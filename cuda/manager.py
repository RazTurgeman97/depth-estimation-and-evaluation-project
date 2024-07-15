#!/usr/bin/env python3

# @author Jesus Alvarez <sw-cuda-installer@nvidia.com>

"""Container scripts template injector and pipeline trigger."""

#
# !! IMPORTANT !!
#
# Editors of this file should use https://github.com/python/black for auto formatting.
#

# >>> REALLY IMPORTANT NOTICE ABOUT DEPENDENCIES... <<<
#
# 1. Dependency handling is done in two places
#
#    a: Dependencies will be added automatically to the kitmaker trigger builder images
#
# 2. Gitlab pipeline "prepare" stage.
#
#    a: This needs to be updated manually at the moment, to do this go to
#    https://gitlab-master.nvidia.com/cuda-installer/cuda/-/pipelines/new and set the variable to "REBUILD_BUILDER=true" and
#    run it. This will run the gitlab builder image rebuild to include the new dependencies.
#
#    It is also possible to trigger builder rebuild with manager.py
#
import re
import os
import pathlib
import logging
import logging.config

# import shutil

import sys

import json

from jinja2 import Environment

from plumbum import cli
from plumbum.cmd import rm  # type: ignore

import yaml
import glom
import docker

import requests
from retry import retry

from typing import Any

from config import *
from error import *
from utils import *
from shipitdata import *
from dotdict import *


class Manager(cli.Application):
    """CUDA CI Manager"""

    PROGNAME: str = "manager.py"
    VERSION: str = "0.0.1"

    manifest: dict[str, dict[str, str]] = {}
    ci = None

    manifest_path: Any = cli.SwitchAttr(
        "--manifest",
        str,
        excludes=["--shipit-uuid"],
        help="Select a manifest to use.",
    )

    shipit_uuid: Any = cli.SwitchAttr(
        "--shipit-uuid",
        str,
        excludes=["--manifest"],
        help="Shipit UUID used to build release candidates (internal)",
    )

    def _load_manifest_yaml(self):
        log.debug(f"Loading manifest: {self.manifest_path}")
        with open(self.manifest_path, "r") as f:
            self.manifest = yaml.load(f, yaml.Loader)

    def load_ci_yaml(self):
        with open(".gitlab-ci.yml", "r") as f:
            self.ci = yaml.load(f, yaml.Loader)

    def _load_app_config(self):
        with open("manager-config.yaml", "r") as f:
            logging.config.dictConfig(yaml.safe_load(f.read())["logging"])

    # Get data from a object by dotted path. Example "cuda."v10.0".cuda_requires"
    def get_data(
        self, obj: Any, *path: str, can_skip: bool = False
    ) -> dict[str, dict[str, str]]:
        data: Any = {}
        try:
            data = glom.glom(obj, glom.Path(*path))
        except glom.PathAccessError:
            if can_skip:
                return {}
            # raise glom.PathAccessError
            log.error(f'get_data path: "{path}" not found!')
        return data

    def main(self):
        self._load_app_config()
        if not self.nested_command:  # will be ``None`` if no sub-command follows
            log.fatal("No subcommand given!")
            print()
            self.help()
            return 1
        elif len(self.nested_command[1]) < 2 and any(
            "generate" in arg for arg in self.nested_command[1]
        ):
            log.error(
                "Subcommand 'generate' missing  required arguments! use 'generate --help'"
            )
            return 1
        elif not any(halp in self.nested_command[1] for halp in ["-h", "--help"]):
            log.info("cuda ci manager start")
            if not self.shipit_uuid and self.manifest_path:
                self._load_manifest_yaml()


@Manager.subcommand("trigger")  # type: ignore
class ManagerTrigger(Manager):
    DESCRIPTION = "Trigger for changes."

    trigger_all: bool = False
    trigger_explicit: List[str] = []

    key: str = ""
    pipeline_name: str = "default"

    parent: Manager

    CI_API_V4_URL = "https://gitlab-master.nvidia.com/api/v4"
    CI_PROJECT_ID = 12064

    dry_run: Any = cli.Flag(
        ["-n", "--dry-run"], help="Show output but don't make any changes."
    )

    no_test: Any = cli.Flag(["--no-test"], help="Don't run smoke tests")

    no_scan: Any = cli.Flag(["--no-scan"], help="Don't run security scans")

    no_push: Any = cli.Flag(["--no-push"], help="Don't push images to the registries")

    no_cache: Any = cli.Flag(["--no-cache"], help="Don't use the image cache during build")

    rebuildb: Any = cli.Flag(
        ["--rebuild-builder"],
        help="Force rebuild of the builder image used to build the cuda images.",
    )

    distro: Any = cli.SwitchAttr(
        "--os-name",
        str,
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="The distro name without version information",
        default=None,
    )

    distro_version: Any = cli.SwitchAttr(
        "--os-version",
        str,
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="The distro version",
        default=None,
    )

    release_label: Any = cli.SwitchAttr(
        "--release-label",
        str,
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="The cuda release label. Example: 11.2.0",
        default=None,
    )

    arch: Any = cli.SwitchAttr(
        "--arch",
        cli.Set("x86_64", "arm64", case_sensitive=False),
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="Generate container scripts for a particular architecture",
    )

    candidate_number: Any = cli.SwitchAttr(
        "--candidate-number",
        str,
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="The CUDA release candidate number",
        default=None,
    )

    candidate_url: Any = cli.SwitchAttr(
        "--candidate-url",
        str,
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="The CUDA release candidate url",
        default=None,
    )

    webhook_url: Any = cli.SwitchAttr(
        "--webhook-url",
        str,
        group="Targeted",
        excludes=["--manifest", "--trigger-override"],
        help="The url to POST to when the job is done. POST will include a list of tags pushed",
        default=None,
    )

    branch: Any = cli.SwitchAttr(
        "--branch",
        str,
        group="Targeted",
        help="The branch to trigger against on gitlab.",
        default=None,
    )

    trigger_override: Any = cli.SwitchAttr(
        "--trigger-override",
        str,
        excludes=["--shipit-uuid"],
        help="Override triggering from gitlab with a variable",
        default=None,
    )

    flavor: Any = cli.SwitchAttr(
        "--flavor",
        str,
        help="The container configuration to build (limited support).",
        default="default",
    )

    def ci_pipeline_by_name(self, name):
        rgx = re.compile(rf"^\s+- if: '\$([\w\._]*{name}[\w\._]*)\s+==\s.true.'$")
        ci_vars = []
        with open(".gitlab-ci.yml", "r") as fp:
            for _, line in enumerate(fp):
                match = rgx.match(line)
                if match:
                    ci_vars.append(match.groups(0)[0])
        return ci_vars

    def ci_pipelines(
        self,
        cuda_version,
        distro,
        distro_version,
    ):
        """Returns a list of pipelines extracted from the gitlab-ci.yml

        Iterates .gitlab-ci.yml line by line looking for a match on the pipeline variable.

        For example:

            - if: '$ubuntu20_04_11_3_1 == "true"'

        Every pipeline has this variable defined, and that is used with the gitlab trigger api to trigger explicit
        pipelines.

        Returns a list of pipeline variables to pass to the gitlab API.

        All arguments to this function can be None, in that case all of the pipelines are returned.
        """
        if not cuda_version:
            cuda_version = r"(\d{1,2}_\d{1,2}_?\d?)"
        if not distro:
            distro = r"(ubuntu\d{2}_\d{2})?(?(2)|([a-z]*\d))"
        elif not distro_version:
            distro = rf"({distro}\d)"
            if "ubuntu" in distro:
                distro = r"(ubuntu\d{2}_\d{2})"
        else:
            distro = f"{distro}{distro_version}"

        rgx_temp = rf"^\s+- if: '\$({distro}_{cuda_version})\s+==\s.true.'$"
        #  log.debug(f"regex_matcher: {rgx_temp}")
        rgx = re.compile(rgx_temp)
        ci_vars = []
        with open(".gitlab-ci.yml", "r") as fp:
            for _, line in enumerate(fp):
                match = rgx.match(line)
                if match:
                    ci_vars.append(match.groups(0)[0])
        return ci_vars

    def get_cuda_version_from_trigger(self, trigger: str) -> str:
        rgx = re.compile(r".*cuda-?([\d\.]+).*$")
        match = rgx.match(trigger)
        if (match := rgx.match(trigger)) is not None:
            return match.group(1)
        else:
            log.info(f"Cuda version not found in trigger!")
        return ""

    def get_pipeline_name_from_trigger(self, trigger: str) -> str:
        rgx = re.compile(r".*name:(\w+)$")
        if (match := rgx.match(trigger)) is not None:
            return match.group(1)
        return ""

    def get_distro_version_from_trigger(self, trigger: str) -> str:
        rgx = re.compile(r".*cuda([\d\.]+).*$")
        match = rgx.match(trigger)
        if match is not None:
            return match.group(1)
        else:
            log.warning(f"Could not extract version from trigger: '{trigger}'!")
        return ""

    def check_explicit_trigger(self):
        """Checks for a pipeline trigger command and builds a list of pipelines to trigger."""
        if not re.search(
            r"^(?:[cuda]+[\d\.]*(?:[_a-z0-9]*)?,?)+$", self.trigger_override
        ):
            raise Exception(
                (
                    "Regex match for trigger override failed! "
                    "Allowed format is 'cuda<version>(_<distro_with_version>)[,...]' "
                    "ex: 'cuda11.0.3' or 'cuda10.2_centos8`"
                )
            )

    def populate_triggers(self) -> bool:
        """Builds a list of gitlab pipelines to trigger.

        The list is stored is trigger_explicit member variable.

        Returns True if pipelines have been found matching the trigger command.
        """
        # 20:22 Thu May 19 2022: commented out because it prevented using explicit trigger with distro 'rockylinux8'
        # self.check_explicit_trigger()
        pipeline: str = self.trigger_override
        if "all" in pipeline:
            log.info("Triggering ALL of the jobs!")
            self.trigger_all = True
            return True

        jobs = []
        jobs.append(pipeline)
        log.debug(f"jobs: {jobs}")

        if "," in pipeline:
            jobs = [x.strip() for x in pipeline.split(",")]

        for job in jobs:
            version = self.get_cuda_version_from_trigger(job)
            if not version:
                self.pipeline_name = self.get_pipeline_name_from_trigger(job)

            log.debug("cuda_version: %s" % version)
            log.debug("pipeline_name: %s" % self.pipeline_name)

            self.key = f"cuda_v{version}"
            if self.pipeline_name != "default":
                self.key = f"cuda_v{version}_{self.pipeline_name}"

            distro_in_job = job
            if "_" in job:
                distro_in_job = job.split("_")[1]

            distro = next(
                (
                    d.distro
                    for d in supported_platforms.list
                    if d.distro in distro_in_job
                ),
                None,
            )
            distro_version = next(
                (
                    d.version
                    for d in supported_platforms.list
                    if d.version in distro_in_job
                ),
                None,
            )

            # Any or all of the variables passed to this function can be None
            for cvar in self.ci_pipelines(version, distro, distro_version):
                if self.pipeline_name and not "default" in self.pipeline_name:
                    pipeline_vars = self.ci_pipeline_by_name(self.pipeline_name)
                else:
                    pipeline_vars = self.ci_pipelines(version, distro, distro_version)

                for cvar in pipeline_vars:
                    if not cvar in self.trigger_explicit:
                        log.info("Triggering '%s'", cvar)
                        self.trigger_explicit.append(cvar)
        return True

    def kickoff(self):
        url = os.getenv("CI_API_V4_URL") or self.CI_API_V4_URL
        project_id = os.getenv("CI_PROJECT_ID") or self.CI_PROJECT_ID
        dry_run = os.getenv("DRY_RUN") or self.dry_run
        no_test = os.getenv("NO_TEST") or self.no_test
        no_scan = os.getenv("NO_SCAN") or self.no_scan
        no_push = os.getenv("NO_PUSH") or self.no_push
        no_cache = os.getenv("NO_CACHE") or self.no_cache
        rebuildb = os.getenv("REBUILD_BUILDER") or self.rebuildb
        token = os.getenv("CI_JOB_TOKEN")
        if not token:
            log.warning("CI_JOB_TOKEN is unset!")
        ref = os.getenv("CI_COMMIT_REF_NAME") or self.branch
        payload = {"token": token, "ref": ref, "variables[TRIGGER]": "true"}
        if self.trigger_all:
            payload["variables[all]"] = "true"
        elif self.trigger_explicit:
            for job in self.trigger_explicit:
                payload[f"variables[{job}]"] = "true"
        if no_scan:
            payload[f"variables[NO_SCAN]"] = "true"
        if no_test:
            payload[f"variables[NO_TEST]"] = "true"
        if no_push:
            payload[f"variables[NO_PUSH]"] = "true"
        if no_cache:
            payload[f"variables[NO_CACHE]"] = "true"
        if rebuildb:
            payload[f"variables[REBUILD_BUILDER]"] = "true"
        if self.flavor:
            payload[f"variables[FLAVOR]"] = self.flavor
        final_url = f"{url}/projects/{project_id}/trigger/pipeline"
        log.info("url %s", final_url)
        log.info("payload %s", payload)
        if not dry_run:
            r = requests.post(final_url, data=payload)
            log.info("response status code %s", r.status_code)
            log.info("response body %s", r.json())
        else:
            log.info("In dry-run mode so not making gitlab trigger POST")

    def kickoff_from_kitmaker(self):
        url = os.getenv("CI_API_V4_URL") or self.CI_API_V4_URL
        project_id = os.getenv("CI_PROJECT_ID") or self.CI_PROJECT_ID
        dry_run = os.getenv("DRY_RUN") or self.dry_run
        no_test = os.getenv("NO_TEST") or self.no_test
        no_scan = os.getenv("NO_SCAN") or self.no_scan
        no_push = os.getenv("NO_PUSH") or self.no_push
        no_cache = os.getenv("NO_CACHE") or self.no_cache
        token = os.getenv("CI_JOB_TOKEN")
        if not token:
            log.warning("CI_JOB_TOKEN is unset!")
        ref = os.getenv("CI_COMMIT_REF_NAME") or self.branch
        payload = {"token": token, "ref": ref, "variables[KITMAKER]": "true"}
        if no_scan:
            payload[f"variables[NO_SCAN]"] = "true"
        if no_test:
            payload[f"variables[NO_TEST]"] = "true"
        if no_push:
            payload[f"variables[NO_PUSH]"] = "true"
        if no_cache:
            payload[f"variables[NO_CACHE]"] = "true"
        if "l4t" in self.flavor:  # type: ignore
            # FIXME: HACK until these scripts can be made to have proper container image "flavor" support
            payload[f"variables[UBUNTU20_04_L4T]"] = "true"
        if self.flavor:
            payload[f"variables[FLAVOR]"] = self.flavor
        payload[f"variables[TRIGGER]"] = "true"
        payload[f"variables[RELEASE_LABEL]"] = self.release_label
        payload[f"variables[IMAGE_TAG_SUFFIX]"] = f"-{self.candidate_number}"
        payload[f"variables[CANDIDATE_URL]"] = self.candidate_url
        payload[f"variables[WEBHOOK_URL]"] = self.webhook_url

        final_url = f"{url}/projects/{project_id}/trigger/pipeline"
        log.info("url %s", final_url)
        masked_payload = payload.copy()
        masked_payload["token"] = "[ MASKED ]"
        log.info("payload %s", masked_payload)

        if not dry_run:
            r = requests.post(final_url, data=payload)
            log.info("response status code %s", r.status_code)
            log.info("response body %s", r.json())
            if r.status_code == 404:
                log.error("response is 404")
                sys.exit(1)
        else:
            log.info("In dry-run mode so not making gitlab trigger POST")

    def main(self):
        if self.dry_run:
            log.info("Dryrun mode enabled. Not making changes")

        if self.parent.shipit_uuid:
            self.shipit = ShipitData(self.parent.shipit_uuid)
            # Make sure all of our arguments are present
            if any(
                [
                    not i
                    for i in [
                        self.release_label,
                        self.candidate_number,
                        self.candidate_url,
                        self.webhook_url,
                        self.branch,
                    ]
                ]
            ):
                # Plumbum doesn't allow this check
                log.error(
                    """Missing arguments (one or all): ["--release_label", "--candidate-number", "--candidate_url", "--webhook-url", "--branch"]"""
                )
                sys.exit(1)
            log.debug("Triggering gitlab kitmaker pipeline using shipit source")
            self.kickoff_from_kitmaker()
        else:
            if self.populate_triggers():
                self.kickoff()


@Manager.subcommand("push")  # type: ignore
class ManagerContainerPush(Manager):
    DESCRIPTION = (
        "Login and push to the container registries.\n"
        "Use either --image-name, --os-name, --os-version, --cuda-version 'to push images' or --readme 'to push readmes'."
    )

    parent: Manager

    dry_run = cli.Flag(["-n", "--dry-run"], help="Show output but don't do anything!")

    image_name: Any = cli.SwitchAttr(
        "--image-name",
        str,
        excludes=["--readme"],
        help="The image name to tag",
        default="",
    )

    distro: Any = cli.SwitchAttr(
        "--os-name",
        str,
        help="The distro to use",
        default=None,
    )

    distro_version: Any = cli.SwitchAttr(
        "--os-version",
        str,
        help="The distro version",
        default=None,
    )

    cuda_version: Any = cli.SwitchAttr(
        "--cuda-version",
        str,
        help="The cuda version to use. Example: '10.1'",
        default=None,
    )

    image_tag_suffix: Any = cli.SwitchAttr(
        "--tag-suffix",
        str,
        help="The suffix to append to the tag name. Example 10.1-base-centos6<suffix>",
        default="",
    )

    pipeline_name: Any = cli.SwitchAttr(
        "--pipeline-name",
        str,
        help="The name of the pipeline the deploy is coming from",
    )

    tag_manifest: Any = cli.SwitchAttr(
        "--tag-manifest",
        str,
        help="A list of tags to push",
    )

    readme: Any = cli.Flag(
        "--readme",
        help="Path to the README.md",
    )

    flavor: Any = cli.SwitchAttr(
        "--flavor",
        str,
        help="The container configuration to build (limited support).",
        default="default",
    )

    client = None
    repos = []
    repos_dict = {}
    tags = []
    key = ""
    push_repos: dict[str, dict[str, str]] = {}
    target_repos = []
    repo_creds = {}

    def setup_repos(self):
        # Regular pipeines use this:
        self.push_repos = self.get_data(self.parent.manifest, "push_repos")
        self.target_repos = self.get_data(self.parent.manifest, self.key, "push_repos")

        excluded_repos = self.get_data(
            self.parent.manifest,
            self.key,
            f"{self.distro}{self.distro_version}",
            "exclude_repos",
            can_skip=True,
        )

        for repo, metadata in self.push_repos.items():
            if repo not in self.target_repos:
                continue
            if "gitlab-master" in repo:
                # Images have already been pushed to gitlab by this point
                log.debug(f"Skipping push to {repo}")
                continue
            if metadata.get("only_if", False) and not os.getenv(metadata["only_if"]):
                log.info("repo: '%s' only_if requirement not satisfied", repo)
                continue
            if self.push_repos and repo not in self.push_repos:
                log.info("repo: '%s' is excluded for this image", repo)
                continue
            if excluded_repos and repo in excluded_repos:
                log.info("repo: '%s' is excluded for this image", repo)
                continue
            user = os.getenv(metadata["user"])
            if not user:
                user = metadata["user"]
            passwd = os.getenv(metadata["pass"])
            if not passwd:
                passwd = metadata["pass"]
            registry = metadata["image_name"]
            self.repo_creds[registry] = {"user": user, "pass": passwd}
            self.repos.append(registry)
        if not self.repos:
            log.fatal(
                "Could not retrieve container image repo credentials. Environment not set?"
            )
            sys.exit(1)

    @retry(
        (ImagePushRetry),
        tries=HTTP_RETRY_ATTEMPTS,
        delay=HTTP_RETRY_WAIT_SECS,
        logger=log,
    )
    def push_images(self):
        with open(self.tag_manifest) as f:
            tags = f.readlines()
        stags = [x.strip() for x in tags]
        for tag in stags:
            if not tag:
                continue
            log.info("Processing image: %s:%s", self.image_name, tag)
            for repo in self.repos:
                log.info("COPYING to: %s:%s", repo, tag)
                if self.dry_run:
                    log.debug("dry-run; not copying")
                    continue
                out = shellcmd(
                    "skopeo",
                    (
                        "copy",
                        "--all",
                        "--src-creds",
                        "{}:{}".format("gitlab-ci-token", os.getenv("CI_JOB_TOKEN")),
                        "--dest-creds",
                        "{}:{}".format(
                            self.repo_creds[repo]["user"],
                            self.repo_creds[repo]["pass"],
                        ),
                        f"docker://{self.image_name}:{tag}",
                        f"docker://{repo}:{tag}",
                        "--retry-times",
                        "2",
                        #  "--debug",
                    ),
                )
                if out.returncode == 0:
                    log.info("Copy was successful")
                else:
                    raise ImagePushRetry()

    def push_readmes(self):
        if self.dry_run:
            log.debug(
                f"dry-run mode: otherwise; docker pushrm could happen for -> {self.repos_dict['docker.io']}"
            )
        else:
            for readme, repo in self.repos_dict["docker.io"].items():
                # docker pushrm
                result = shellcmd(
                    "docker",
                    ("pushrm", "-f", f"doc/{readme}", f"{repo}"),
                    printOutput=False,
                )
                if result.returncode > 0:
                    log.error(result.stderr)
                    log.error("Docker pushrm was unsuccessful for %s", repo)
                else:
                    log.info("Docker pushrm was successful for %s", repo)

    def main(self):
        log.debug("dry-run: %s", self.dry_run)
        if self.readme:
            self.key = f"push_repos"
            self.push_readmes()
        else:
            self.key = f"cuda_v{self.cuda_version}"
            if self.pipeline_name:
                self.key = f"cuda_v{self.cuda_version}_{self.pipeline_name}"
            self.client = docker.DockerClient(
                base_url="unix://var/run/docker.sock", timeout=600
            )
            self.setup_repos()
            self.push_images()
        log.info("Done")


@Manager.subcommand("generate")  # type: ignore
class ManagerGenerate(Manager):
    DESCRIPTION = "Generate Dockerfiles from templates."

    parent: Manager

    cuda = {}
    output_path = {}  # The product of parsing the input templates

    dist_base_path = None  # pathlib object. The parent "base" path of output_path.
    output_manifest_path = None  # pathlib object. The path to save the shipit manifest.

    key: str = ""
    cuda_version_is_release_label = False
    cuda_version_regex = re.compile(r"cuda_v([\d\.]+)(?:_(\w+))?$")

    product_name: str = ""
    candidate_number: str = ""

    shipitdata = None

    # ( ͡° ͜ʖ ͡°)
    tegra: bool = False

    # RIP
    end_of_life = False

    template_env: Any = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
    )

    generate_ci: Any = cli.Flag(
        ["--ci"],
        help="Generate the gitlab pipelines only.",
    )

    generate_all: Any = cli.Flag(
        ["--all"],
        help="Generate all of the templates.",
    )

    generate_readme: Any = cli.Flag(
        ["--readme"],
        help="Generate all readmes.",
    )

    generate_tag: Any = cli.Flag(
        ["--tags"],
        help="Generate all supported and unsupported tag lists.",
    )

    distro: Any = cli.SwitchAttr(
        "--os-name",
        str,
        group="Targeted",
        excludes=["--all", "--readme", "--tags"],
        help="The distro to use.",
        default=None,
    )

    distro_version: Any = cli.SwitchAttr(
        "--os-version",
        str,
        group="Targeted",
        excludes=["--all", "--readme", "--tags"],
        help="The distro version",
        default=None,
    )

    cuda_version: Any = cli.SwitchAttr(
        "--cuda-version",
        str,
        excludes=["--all", "--readme", "--tags"],
        group="Targeted",
        help="[DEPRECATED for newer cuda versions!] The cuda version to use. Example: '11.2'",
        default=None,
    )

    release_label: Any = cli.SwitchAttr(
        "--release-label",
        str,
        excludes=["--readme", "--tags"],
        group="Targeted",
        help="The cuda version to use. Example: '11.2.0'",
        default=None,
    )

    pipeline_name: Any = cli.SwitchAttr(
        "--pipeline-name",
        str,
        excludes=["--all", "--readme", "--tags"],
        group="Targeted",
        help="Use a pipeline name for manifest matching.",
        default="default",
    )

    flavor: Any = cli.SwitchAttr(
        "--flavor",
        str,
        help="Identifier passed to template context.",
    )

    #
    # WAR ONLY USED FOR L4T and will be removed in the future
    #
    cudnn_json_path: Any = cli.SwitchAttr(
        "--cudnn-json-path",
        str,
        group="L4T",
        help="File path to json encoded file containing cudnn package metadata.",
    )

    def cudnn_versions(self, arch: str) -> list[str]:
        obj = []
        for k, v in self.cuda[arch]["components"].items():
            if k.startswith("cudnn") and v:
                obj.append(k)
        return obj

    def matched(self, key):
        match = self.cuda_version_regex.match(key)
        if match:
            return match

    # extracts arbitrary keys and inserts them into the templating context
    def extract_keys(self, val, arch=None):
        rgx = re.compile(r"^v\d+\.\d")
        for k, v in val.items():
            if rgx.match(k):
                # Do not copy cuda version keys
                continue
            # These top level keys should be ignored since they are processed elsewhere
            data = self.parent.manifest
            if self.shipitdata:
                data = self.shipitdata.shipit_manifest
            if k in [
                "exclude_repos",
                "components",
                *self.arches,
                *supported_distro_list_by_cuda_version(
                    data,
                    self.cuda_version,
                ),
            ]:
                continue
            newv = v
            if isinstance(v, str):
                newv = v.strip()
            if arch:
                self.cuda[arch][k] = newv
            else:
                self.cuda[k] = newv

    # For cudnn templates, we need a custom template context
    def output_cudnn_template(self, cudnn_version_name, input_template, output_path):
        new_ctx = {
            "cudnn": self.cuda["cudnn"],
            "version": self.cuda["version"],
            "image_tag_suffix": self.cuda["image_tag_suffix"],
            "os": self.cuda["os"],
        }
        for arch in self.arches:
            larch = arch
            if self.tegra:
                larch = arches.tegra.container_arch
            if not cudnn_version_name in self.cuda[larch]["components"]:
                continue
            cudnn_manifest = self.cuda[larch]["components"][cudnn_version_name]
            if cudnn_manifest:
                if "source" in cudnn_manifest:
                    cudnn_manifest["basename"] = os.path.basename(
                        cudnn_manifest["source"]
                    )
                    cudnn_manifest["dev"]["basename"] = os.path.basename(
                        cudnn_manifest["dev"]["source"]
                    )
                new_ctx[larch] = {}
                new_ctx[larch]["cudnn"] = cudnn_manifest

        log.debug(f"cudnn template context {pp(new_ctx, output=False)}")
        self.output_template(
            input_template=input_template, output_path=output_path, ctx=new_ctx
        )

    def output_template(self, input_template, output_path, ctx=None):
        ctx = ctx if ctx is not None else self.cuda

        def write_template(arch=None):
            with open(input_template) as f:
                log.debug("Processing template %s", input_template)
                new_output_path = pathlib.Path(output_path)
                extension = ".j2"
                name = input_template.name
                if "dockerfile" in input_template.name.lower():
                    new_filename = "Dockerfile"
                elif ".jinja" in str(input_template):
                    extension = ".jinja"
                    new_filename = (
                        name[: -len(extension)] if name.endswith(extension) else name
                    )
                else:
                    new_filename = (
                        name[len("base-") : -len(extension)]
                        if name.startswith("base-") and name.endswith(extension)
                        else name
                    )
                if arch:
                    new_filename += f"-{arch}"
                template = self.template_env.from_string(f.read())
                if not new_output_path.exists():
                    log.debug(f"Creating {new_output_path}")
                    new_output_path.mkdir(parents=True)
                log.info(f"Writing {new_output_path}/{new_filename}")
                with open(f"{new_output_path}/{new_filename}", "w") as f2:
                    f2.write(template.render(cuda=ctx))

        if "cuda.repo" in input_template.as_posix():
            for arch in self.arches:
                ctx["target_arch"] = arches.get(arch).repo_name.rpm
                if self.tegra:
                    arch = "tegra"
                write_template(arches.get(arch).common_name)
        else:
            write_template()

    def prepare_context(self):
        conf = self.parent.manifest
        if self.release_label:
            major = self.release_label.split(".")[0]
            minor = self.release_label.split(".")[1]
        else:
            major = self.cuda_version.split(".")[0]
            minor = self.cuda_version.split(".")[1]

        self.end_of_life = self.get_data(conf, self.key, "end_of_life", can_skip=True)
        self.image_tag_suffix = self.get_data(
            conf,
            self.key,
            f"{self.distro}{self.distro_version}",
            "image_tag_suffix",
            can_skip=True,
        )
        if not self.image_tag_suffix:
            self.image_tag_suffix = ""

        larch = self.arches
        if self.tegra:
            larch = arches.tegra.container_arch

        # The templating context. This data structure is used to fill the templates.
        self.cuda = {
            "supported_arches": arches,
            "flavor": self.flavor,
            "version": {
                "release_label": self.cuda_version,
                "major": major,
                "minor": minor,
                "major_minor": f"{major}.{minor}",
            },
            "arches": larch,
            "os": {"distro": self.distro, "version": self.distro_version},
            "image_tag_suffix": self.image_tag_suffix,
            "end_of_life": self.end_of_life,
        }

        self.extract_keys(
            self.get_data(
                conf,
                self.key,
                f"{self.distro}{self.distro_version}",
            )
        )

        for arch in self.arches:
            if arch in arches:
                translated_arch = arches.get(arch).container_arch
            elif self.tegra:
                translated_arch = arches.tegra.container_arch
            if self.tegra:
                larch = arches.tegra.common_name
            self.cuda[translated_arch] = {}
            # Only set in version < 11.0
            self.cuda["version"]["build_version"] = self.get_data(
                conf,
                self.key,
                "build_version",
                can_skip=True,
            )
            self.cuda[translated_arch]["components"] = self.get_data(
                conf,
                self.key,
                f"{self.distro}{self.distro_version}",
                arch,
                "components",
            )
            self.extract_keys(
                self.get_data(
                    conf,
                    self.key,
                    f"{self.distro}{self.distro_version}",
                    arch,
                ),
                arch=translated_arch,
            )

        log.debug(f"template context {pp(self.cuda, output=False)}")

    def generate_cudnn_scripts(self, base_image, input_template):
        for arch in self.arches:
            larch = arch
            if self.tegra:
                larch = arches.tegra.container_arch
            for pkg in self.cudnn_versions(larch):
                if not "cudnn" in self.cuda:
                    self.cuda["cudnn"] = {}
                self.cuda["cudnn"]["target"] = base_image
                self.output_cudnn_template(
                    cudnn_version_name=pkg,
                    input_template=pathlib.Path(input_template),
                    output_path=pathlib.Path(f"{self.output_path}/{base_image}/{pkg}"),
                )

    def generate_containerscripts(self):
        for img in ["base", "devel", "runtime"]:
            self.cuda["target"] = img

            globber = f"*"
            if "legacy" in self.cuda["template_path"]:
                temp_path = pathlib.Path(self.cuda["template_path"], img)
                cudnn_template_path = pathlib.Path(
                    self.cuda["template_path"], f"cudnn/Dockerfile.jinja"
                )
                input_template = f"{temp_path}/Dockerfile.jinja"
            else:
                temp_path = pathlib.Path(self.cuda["template_path"])
                input_template = pathlib.Path(temp_path, f"{img}-dockerfile.j2")
                cudnn_template_path = pathlib.Path(temp_path, "cudnn-dockerfile.j2")
                globber = f"{img}-*"

            log.debug(
                "template_path: %s, output_path: %s",
                temp_path,
                self.output_path,
            )

            self.output_template(
                input_template=pathlib.Path(input_template),
                output_path=pathlib.Path(f"{self.output_path}/{img}"),
            )

            # copy files
            log.debug(f"temp_path: {temp_path} img: {img}")
            for filename in pathlib.Path(temp_path).glob(globber):
                log.info(f"have template: {filename}")
                if "dockerfile" in filename.name.lower():
                    continue
                log.debug("Checking %s", filename)
                if any(f in filename.name for f in [".j2", ".jinja"]):
                    self.output_template(filename, f"{self.output_path}/{img}")

            # cudnn image
            if "base" not in img:
                self.generate_cudnn_scripts(img, cudnn_template_path)

    def generate_gitlab_pipelines(self):

        manifest = self.parent.manifest
        ctx = {
            "supported_distro_matrix": supported_platforms,
            "moby_buildkit_version": MOBY_BUILDKIT_VERSION,
            "manifest_path": self.parent.manifest_path,
        }

        def get_cudnn_components(key, distro, arch):
            comps = {}
            for comp, val in manifest[key][distro][arch]["components"].items():
                if "cudnn" in comp and val:
                    comps[comp] = {}
                    comps[comp]["version"] = val["version"]
            return comps

        for k, _ in manifest.items():
            if (match := self.matched(k)) is None:
                log.debug("No match for %s" % k)
                continue

            log.info("Adding pipeline '%s'" % k)
            cuda_version = match.group(1)
            if (pipeline_name := match.group(2)) is None:
                pipeline_name = "default"
            log.debug("matched cuda_version: %s" % cuda_version)
            log.debug("matched pipeline_name: %s" % pipeline_name)

            if cuda_version not in ctx:
                ctx[cuda_version] = {}
            ctx[cuda_version][pipeline_name] = {}
            ctx[cuda_version][pipeline_name][
                "cuda_version_yaml_safe"
            ] = cuda_version.replace(".", "_")

            key = f"cuda_v{cuda_version}"
            if pipeline_name and pipeline_name != "default":
                key = f"cuda_v{cuda_version}_{pipeline_name}"

            ctx[cuda_version][pipeline_name]["dist_base_path"] = self.get_data(
                manifest, key, "dist_base_path"
            )
            ctx[cuda_version][pipeline_name]["pipeline_name"] = self.pipeline_name

            for distro, _ in manifest[key].items():
                dmrgx = re.compile(r"(?P<name>[a-zA-Z]+)(?P<version>[\d\.]+)$")
                if (dm := dmrgx.match(distro)) is None:
                    continue
                if not "distros" in ctx[cuda_version][pipeline_name]:
                    ctx[cuda_version][pipeline_name]["distros"] = {}
                ctx[cuda_version][pipeline_name]["distros"][distro] = {}
                ctx[cuda_version][pipeline_name]["distros"][distro]["name"] = dm.group(
                    "name"
                )
                ctx[cuda_version][pipeline_name]["distros"][distro][
                    "version"
                ] = dm.group("version")
                ctx[cuda_version][pipeline_name]["distros"][distro][
                    "yaml_safe"
                ] = distro.replace(".", "_")
                image_tag_suffix = self.get_data(
                    manifest, key, distro, "image_tag_suffix", can_skip=True
                )
                ctx[cuda_version][pipeline_name]["distros"][distro][
                    "image_tag_suffix"
                ] = ""

                if image_tag_suffix:
                    ctx[cuda_version][pipeline_name]["distros"][distro][
                        "image_tag_suffix"
                    ] = image_tag_suffix

                ctx[cuda_version][pipeline_name]["distros"][distro]["arches"] = []

                for arch, _ in manifest[key][distro].items():
                    if arch not in ["arm64", "x86_64"]:
                        continue
                    no_os_suffix = self.get_data(
                        manifest, key, distro, "no_os_suffix", can_skip=True
                    )
                    ctx[cuda_version][pipeline_name]["image_name"] = self.get_data(
                        manifest, key, "image_name"
                    )

                    if (
                        "no_os_suffix"
                        not in ctx[cuda_version][pipeline_name]["distros"][distro]
                    ):
                        ctx[cuda_version][pipeline_name]["distros"][distro][
                            "no_os_suffix"
                        ] = {}

                    ctx[cuda_version][pipeline_name]["distros"][distro][
                        "no_os_suffix"
                    ] = (True if no_os_suffix else False)
                    ctx[cuda_version][pipeline_name]["distros"][distro][
                        "arches"
                    ].append(arch)

                    if (
                        "cudnn"
                        not in ctx[cuda_version][pipeline_name]["distros"][distro]
                    ):
                        ctx[cuda_version][pipeline_name]["distros"][distro][
                            "cudnn"
                        ] = {}
                    cudnn_comps = get_cudnn_components(key, distro, arch)
                    if cudnn_comps:
                        ctx[cuda_version][pipeline_name]["distros"][distro]["cudnn"][
                            arch
                        ] = cudnn_comps

        log.debug(f"template context: {pp(ctx, output=False)}")
        input_template = pathlib.Path("templates/gitlab/gitlab-ci.yml.jinja")
        with open(input_template) as f:
            log.debug("Processing template %s", input_template)
            output_path = pathlib.Path(".gitlab-ci.yml")
            template = self.template_env.from_string(f.read())
            with open(output_path, "w") as f2:
                f2.write(template.render(cuda=ctx))

    def generate_readmes(self):

        distros = []  # local list variable to hold different distros

        # to capture all release labels and corresponding Dockerfile's paths
        release_info = {}
        cuda_release_info = {}

        manifest = self.parent.manifest
        path = {"manifest_path": self.parent.manifest_path}

        def get_releaseInfo_and_dockerfilePath(path):

            for dirpath, _, files in os.walk(path):
                refPath = dirpath.split("dist/")
                for file in files:
                    if file == "Dockerfile":
                        labels = {}
                        releaseLabel = ""
                        dockerfilePath = os.path.join(refPath[1], file)
                        dockerfilePathList = dockerfilePath.split("/")

                        for value in dockerfilePathList:
                            if re.compile(r"([\d\.]+)").match(value):
                                labels[1] = value
                            if "cudnn" in value:
                                labels[2] = value
                            if value in ("base", "devel", "runtime"):
                                labels[3] = value
                            if re.compile(r"centos*|ubuntu*|ubi*|rocky*").match(value):
                                operating_system = value.split("-")
                                labels[4] = operating_system[0]
                                dotdistro = labels[4]
                                if "ubuntu" in labels[4]:
                                    dotdistro = f"{labels[4][:-2]}.{labels[4][-2:]}"
                                distros.append(dotdistro)

                        for key in sorted(labels.keys()):
                            if not releaseLabel:
                                releaseLabel = releaseLabel + labels[key]
                            else:
                                distrokey = labels[key]
                                if "ubuntu" in labels[key]:
                                    distrokey = f"{labels[key][:-2]}.{labels[key][-2:]}"
                                releaseLabel = releaseLabel + "-" + distrokey

                        # storing all release info in a dictionary variable
                        release_info[dockerfilePath] = releaseLabel

        for key, _ in manifest.items():
            if self.matched(key):
                if self.cuda_version_regex.match(key):
                    path["dist_base_path"] = self.get_data(
                        manifest, key, "dist_base_path"
                    )
                    path["release_label"] = self.get_data(
                        manifest, key, "release_label"
                    )
                    get_releaseInfo_and_dockerfilePath(path["dist_base_path"])
                    break  # to keep data for latest available version only

        dist_path_list = path["dist_base_path"].split("/")

        # to get all unique supported operating system names
        distros = set(distros)
        platforms = DotDict()

        def get_arches_for_platform(os):
            ls = []
            for k in glom.glom(
                self.parent.manifest,
                glom.Path(key, os),
            ):
                if k in ["x86_64", "ppc64le", "arm64"]:
                    ls.append(k)
            return ls

        for OS in distros:
            platforms[OS] = DotDict()
            if "centos" in OS:
                distro = OS.split("centos")
                platforms[OS]["name"] = f"CentOS {distro[1]}"
                platforms[OS]["arches"] = get_arches_for_platform(OS)
            elif "ubuntu" in OS:
                distro = OS.split("ubuntu")
                platforms[OS]["name"] = f"Ubuntu {distro[1]}"
                platforms[OS]["arches"] = get_arches_for_platform(OS)
            elif "rocky" in OS:
                distro = OS.split("rockylinux")
                platforms[OS]["name"] = f"Rockylinux {distro[1]}"
                platforms[OS]["arches"] = get_arches_for_platform(OS)
            else:
                distro = OS.split("ubi")
                platforms[OS]["name"] = f"UBI {distro[1]}"
                platforms[OS]["arches"] = get_arches_for_platform(OS)

        # to help populate the OS types in all readmes in a sorted manner
        for keys in sorted(release_info.keys(), reverse=True):
            cuda_release_info[keys] = release_info[keys]

        # a data structure to manipulate readme template
        readme = {
            "latest_version": dist_path_list[1],
            "release_label": path["release_label"],
            "cuda_release_info": cuda_release_info,
            "platforms": platforms,
        }

        input_template = pathlib.Path("templates/doc/README.md.jinja")
        with open(input_template) as rf:
            output_path = pathlib.Path(f"doc/README.md")
            template = self.template_env.from_string(rf.read())
            log.debug(f"Template context: \n{pp(readme, output=False)}")
            with open(output_path, "w") as wf:
                wf.write(template.render(readme=readme))

    def generate_tags(self):

        tag_list = []  # local list variable to hold tags from dockerhub
        distros_list = []  # local list variable to hold distros from dockerhub
        cuda_releases = {}  # local dict variable for all CUDA releases
        unsupported_release_labels = (
            []
        )  # to grab unsupported CUDA releases from manifest.yaml
        unsupported_distros = []  # to grab unsupported CUDA distros from manifest.yaml
        # for all supported CUDA releases
        supported_cuda_releases = {}
        supported_distros = []

        docker_repo = "docker.io/nvidia/cuda"

        def get_repo_tags(repo):
            return shellcmd(
                "skopeo",
                ("list-tags", f"docker://{repo}"),
                printOutput=False,
            )

        try:
            tag_dict = json.loads(get_repo_tags(docker_repo).stdout)
        except Exception as e:
            log.error("Some problem occurred in getting tags from DockerHub")
            log.error(e)
            sys.exit(1)

        for key in tag_dict.keys():
            if "Tags" in key:
                tag_list = list(tag_dict[key])

        for tags in tag_list:
            if "ubuntu" in tags:
                ubuntu_tags = tags.split("-")
                for tag in ubuntu_tags:
                    if "ubuntu" in tag:
                        distros_list.append(tag)
            elif "centos" in tags:
                centos_tags = tags.split("-")
                distros_list.append(centos_tags[len(centos_tags) - 1])
            elif "ubi" in tags:
                ubi_tags = tags.split("-")
                distros_list.append(ubi_tags[len(ubi_tags) - 1])
            elif "rockylinux" in tags:
                rockylinux_tags = tags.split("-")
                distros_list.append(rockylinux_tags[len(rockylinux_tags) - 1])
        distros_set = set(distros_list)

        manifest = self.parent.manifest

        for key, _ in manifest.items():
            if self.matched(key):
                if self.cuda_version_regex.match(key):
                    new_key = key.split("v")
                    cuda_releases[new_key[1]] = self.get_data(
                        manifest, key, "release_label"
                    )
            if key == "unsupported":
                unsupported_distros = self.get_data(manifest, key, "distros")
                unsupported_release_labels = self.get_data(
                    manifest, key, "release_label"
                )

        for distro in distros_set:
            if distro not in unsupported_distros:
                supported_distros.append(distro)

        for key, value in cuda_releases.items():
            if value not in unsupported_release_labels:
                supported_cuda_releases[key] = cuda_releases[key]

        # update cuda_releases with unsupported release info
        for label in unsupported_release_labels:
            if re.compile(r"([\d\.]+)$").match(str(label)):
                cuda_releases[str(label)] = label
            else:  # to handle special cases like 11.0 RC and 11.0 Update 1
                tag = label.split(" ")
                if len(tag) == 2:
                    cuda_releases[tag[0]] = label
                else:
                    cuda_releases[(tag[0] + "." + tag[len(tag) - 1])] = label

        supported = {
            "cuda_tags": tag_list,
            "supported_distros": sorted(supported_distros, reverse=True),
            "supported_cuda_releases": supported_cuda_releases,
        }

        unsupported = {
            "cuda_tags": tag_list,
            "cuda_releases": cuda_releases,
            "unsupported_distros": sorted(unsupported_distros, reverse=True),
            "unsupported_cuda_releases": unsupported_release_labels,
            "supported_distros": sorted(supported_distros, reverse=True),
        }

        for tag in ("supported", "unsupported"):
            if "unsupported" in tag:
                tags = unsupported
            else:
                tags = supported
            input_template = pathlib.Path(f"templates/doc/{tag}-tags.md.Jinja")
            with open(input_template) as rf:
                log.debug("Processing template %s for %s tags", input_template, tag)
                output_path = pathlib.Path(f"doc/{tag}-tags.md")
                template = self.template_env.from_string(rf.read())
                with open(output_path, "w") as wf:
                    wf.write(template.render(tags=tags))

    def set_output_path(self, target):
        self.output_path = pathlib.Path(
            f"{self.dist_base_path}/{target.replace('.', '')}"
        )
        if not self.parent.shipit_uuid and self.output_path.exists:
            log.info(f"Removing {self.output_path}")
            rm["-rf", self.output_path]()
        log.debug(f"self.output_path: '{self.output_path}' target: '{target}'")
        log.debug(f"Creating {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=False)

    def target_all(self):
        log.debug("Generating all container scripts!")
        rgx = re.compile(
            # use regex101.com to debug with gitlab-ci.yml as the search text
            r"^(?P<distro>[a-zA-Z]*)(?P<distro_version>[\d\.]*)-v(?P<cuda_version>[\d\.]*)(?:-(?!cudnn|test|scan|deploy)(?P<pipeline_name>\w+))?$"
        )
        for ci_job, _ in self.parent.ci.items():
            if (match := rgx.match(ci_job)) is None:
                continue
            self.distro = match.group("distro")
            self.distro_version = match.group("distro_version")
            self.cuda_version = match.group("cuda_version")
            if self.cuda_version.count(".") > 1:
                self.cuda_version_is_release_label = True
            self.pipeline_name = match.group("pipeline_name")

            log.debug("ci_job: '%s'" % ci_job)

            self.key = f"cuda_v{self.release_label}"
            if not self.release_label and self.cuda_version:
                self.key = f"cuda_v{self.cuda_version}"

            if self.pipeline_name:
                self.key = f"cuda_v{self.cuda_version}_{self.pipeline_name}"

            self.dist_base_path = pathlib.Path(
                self.parent.get_data(self.parent.manifest, self.key, "dist_base_path")
            )

            log.debug("dist_base_path: %s" % (self.dist_base_path))
            log.debug(
                "Generating distro: '%s' distro_version: '%s' cuda_version: '%s' release_label: '%s' "
                % (
                    self.distro,
                    self.distro_version,
                    self.cuda_version,
                    self.release_label,
                )
            )
            self.targeted()
            self.cuda_version_is_release_label = False

        if not self.dist_base_path:
            log.error("dist_base_path not set!")
            sys.exit(1)

    def target_all_kitmaker(self):
        log.debug("Generating all container scripts! (for kitmaker)")

        key = f"cuda_v{self.release_label}"
        self.cuda_version = self.release_label
        for k, _ in self.shipitdata.shipit_manifest[key].items():
            if any(x.full_name() in k for x in supported_platforms.list) or "l4t" in k:
                log.debug(f"Working on {k}")
                if "l4t" in k:
                    self.tegra = True
                    self.distro = "l4t"
                    self.distro_version = ""
                else:
                    rgx = re.search(r"(\D*)([\d\.]*)", k)
                    self.distro = rgx.group(1)
                    self.distro_version = rgx.group(2)
                self.cuda_version_is_release_label = True

                log.debug(
                    "Generating distro: '%s' distro_version: '%s' cuda_version: '%s' release_label: '%s' "
                    % (
                        self.distro,
                        self.distro_version,
                        self.cuda_version,
                        self.release_label,
                    )
                )
                self.parent.manifest = self.shipitdata.shipit_manifest
                self.targeted()
                self.cuda_version_is_release_label = False

    def targeted_kitmaker(self):
        log.debug("Generating all container scripts! (for kitmaker)")
        key = f"cuda_v{self.release_label}"
        self.cuda_version = self.release_label
        for k, _ in self.shipitdata.shipit_manifest[key].items():
            if any(x.full_name() in k for x in supported_platforms.list) or "l4t" in k:
                if f"{self.distro}{self.distro_version}" not in k:
                    continue
                log.debug(f"Working on {k}")
                self.cuda_version_is_release_label = True
                log.debug(
                    "Generating distro: '%s' distro_version: '%s' cuda_version: '%s' release_label: '%s' "
                    % (
                        self.distro,
                        self.distro_version,
                        self.cuda_version,
                        self.release_label,
                    )
                )
                self.parent.manifest = self.shipitdata.shipit_manifest
                self.targeted()
                self.cuda_version_is_release_label = False

    def targeted(self):
        self.key = f"cuda_v{self.release_label}"
        if not self.release_label and self.cuda_version:
            self.key = f"cuda_v{self.cuda_version}"
        if self.pipeline_name and self.pipeline_name != "default":
            self.key = f"cuda_v{self.release_label}_{self.pipeline_name}"
        log.debug(f"self.key: {self.key}")
        self.arches = supported_arch_list(
            self.parent.manifest,
            f"{self.distro}{self.distro_version}",
            self.cuda_version,
        )

        log.debug(f"self.arches: {self.arches}")

        self.dist_base_path = pathlib.Path(
            self.parent.get_data(
                self.parent.manifest,
                self.key,
                "dist_base_path",
                can_skip=False,
            )
        )
        log.debug(f"self.dist_base_path: {self.dist_base_path}")
        self.set_output_path(f"{self.distro}{self.distro_version}")
        log.debug(f"self.output_manifest_path: {self.output_manifest_path}")
        self.prepare_context()
        self.generate_containerscripts()

    def main(self):
        if self.parent.shipit_uuid:
            log.debug("Have shippit source, generating manifest and scripts")
            self.dist_base_path = pathlib.Path("kitpick")
            self.shipitdata = ShipitData(self.parent.shipit_uuid)
            self.shipitdata.generate_shipit_manifest(
                self.dist_base_path, self.cudnn_json_path
            )
            if self.generate_all:
                self.target_all_kitmaker()
            else:
                self.targeted_kitmaker()
        else:
            if (self.generate_all and not self.parent.shipit_uuid) or self.generate_ci:
                self.generate_gitlab_pipelines()
            elif not (self.generate_readme or self.generate_tags):
                # Make sure all of our arguments are present
                if any(
                    [
                        not i
                        for i in [
                            self.distro,
                            self.distro_version,
                            self.release_label,
                        ]
                    ]
                ):
                    # Plumbum doesn't allow this check
                    log.error(
                        """Missing arguments (one or all): ["--os", "--os-version", "--release-label"]"""
                    )
                    sys.exit(1)
            if not self.generate_ci:
                self.parent.load_ci_yaml()
                if self.generate_all:
                    self.target_all()
                elif self.generate_readme:
                    self.generate_readmes()
                elif self.generate_tag:
                    log.warning("USING TAGS LIST FROM DOCKER HUB...")
                    log.warning(
                        "THIS COMMAND SHOULD BE USED AFTER TAGS HAVE BEEN PUSHED TO DOCKER HUB!"
                    )
                    self.generate_tags()
                else:
                    self.targeted()
        log.info("Done")


@Manager.subcommand("staging-images")  # type: ignore
class ManagerStaging(Manager):

    DESCRIPTION = "Staging image management"

    repos = [
        "gitlab-master.nvidia.com:5005/cuda-installer/cuda",
        "gitlab-master.nvidia.com:5005/cuda-installer/cuda/cuda-arm64",
        "gitlab-master.nvidia.com:5005/cuda-installer/cuda/l4t-cuda",
        "gitlab-master.nvidia.com:5005/cuda-installer/cuda/release-candidate/cuda",
        "gitlab-master.nvidia.com:5005/cuda-installer/cuda/release-candidate/cuda-arm64",
    ]

    delete_all: Any = cli.Flag(
        ["--delete-all"], help="Delete all of the staging images."
    )

    repo: Any = cli.SwitchAttr(
        "--repo",
        cli.Set(*repos, case_sensitive=False),
        excludes=["--delete-all"],
        group="Targeted",
        help="Delete only from a specific repo.",
    )

    def get_repo_tags(self, repo):
        return shellcmd(
            "skopeo",
            ("list-tags", f"docker://{repo}"),
            printOutput=False,
        )

    def delete_all_tags(self):
        for repo in self.repos:
            self.delete_all_tags_repo(repo)

    @retry(
        (ImageDeleteRetry),
        tries=HTTP_RETRY_ATTEMPTS,
        delay=HTTP_RETRY_WAIT_SECS,
        logger=log,
    )
    def delete_all_tags_repo(self, repo):
        out = self.get_repo_tags(repo)
        if out.returncode > 0:
            log.fatal("Could not use skopeo to gat a list of images!")
            sys.exit(1)
        tags = json.loads(out.stdout)["Tags"]
        for tag in tags:
            log.debug(f"deleting {repo}:{tag}")
            out2 = shellcmd(
                "skopeo",
                ("delete", f"docker://{repo}:{tag}"),
                printOutput=False,
            )
            if out2.returncode > 0:
                log.info(f"deleted {repo}:{tag}")
            else:
                raise ImageDeleteRetry()

    def main(self):
        if self.delete_all:
            self.delete_all_tags()
        elif self.repo:
            self.delete_all_tags_repo(self.repo)
        else:
            log.fatal("No flags defined!")
            print()
            self.help()
            return 1


if __name__ == "__main__":
    Manager.run()
