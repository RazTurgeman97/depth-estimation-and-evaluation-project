from config import *
from retry import retry
from error import *

import os
import yaml
import pathlib
import subprocess
import io
import sys
import json
import re
from typing import NamedTuple


from plumbum import local
from plumbum.cmd import find, cut, sort  # type: ignore
import glom


@retry(
    (ImageRegistryLoginRetry),
    tries=HTTP_RETRY_ATTEMPTS,
    delay=HTTP_RETRY_WAIT_SECS,
    logger=log,
)
def auth_registries(push_repos):
    repos = {}
    for repo, metadata in push_repos.items():
        if metadata.get("only_if", False) and not os.getenv(metadata["only_if"]):
            log.info("repo: '%s' only_if requirement not satisfied", repo)
            continue
        user = os.getenv(metadata["user"])
        if not user:
            user = metadata["user"]
        passwd = os.getenv(metadata["pass"])
        if not passwd:
            passwd = metadata["pass"]
        repos[metadata["registry"]] = {"user": user, "pass": passwd}

    if not repos:
        log.fatal("Could not retrieve registry credentials. Environment not set?")
        sys.exit(1)

    # docker login
    for repo, data in repos.items():
        log.info(f"Logging into {repo}")
        #  log.debug(f"USER: {data['user']} PASS: {data['pass']}")
        result = shellcmd(
            "docker",
            (
                "login",
                repo,
                f"-u" f"{data['user']}",
                f"-p" f"{data['pass']}",
            ),
            printOutput=False,
        )
        if result[0] > 0:
            raise ImageRegistryLoginRetry()
        else:
            log.info(f"Docker login to '{repo}' was successful.")


class ShellCmdReturn(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def shellcmd(bin: str, args: tuple[str, ...], printOutput: bool = True) -> ShellCmdReturn:
    """Run the shell command with specified arguments for skopeo/docker

    args        -- A tuple of arguments.
    printOutput -- If True, the output of the command will be send to the logger.
    """
    if "skopeo" in bin:
        bin_name = local["/usr/bin/skopeo"]
    elif "docker" in bin:
        # find docker
        if pathlib.Path("/usr/local/bin/docker").exists():
            bin_name = local["/usr/local/bin/docker"]
        elif pathlib.Path("/usr/bin/docker").exists():
            bin_name = local["/usr/bin/docker"]
        else:
            raise Exception("Can't find docker client executable!")
    else:
        log.error("%s is not supported by method - shellcmd", bin)
        sys.exit(1)
    out = ""
    err = ""
    # XXX: This can leak secrets!
    #  log.debug(f"Running command: bin: {bin} args: {args}")
    p = bin_name.popen(
        args=args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for line in io.TextIOWrapper(p.stdout, encoding="utf-8"):
        # do something with line
        out += line
        if printOutput:
            log.info(line)
    for line in io.TextIOWrapper(p.stderr, encoding="utf-8"):
        err += line
        if printOutput:
            log.error(line)
    p.communicate()
    log.debug(f"command returncode: {p.returncode}")
    return ShellCmdReturn(p.returncode, out, err)


# Returns a list of packages used in the templates
def template_packages(distro):
    # 21:31 Tue Jul 13 2021 FIXME (jesusa): this func feels like a hack
    log.info(f"current directory: {os.getcwd()}")
    temp_dir = "ubuntu"
    if any(d in distro for d in ["centos", "ubi"]):
        temp_dir = "redhat"
    # sometimes the old ways are the best ways...
    cmd = (
        find[
            f"templates/{temp_dir}/",
            "-type",
            "f",
            "-iname",
            "*.j2",
            "-exec",
            "grep",
            r"^ENV\ NV_.*_VERSION\ .*$",
            "{}",
            ";",
        ]
        | cut["-f", "2", "-d", " "]
        | sort["-u"]
    )
    log.debug(f"command: {cmd}")
    # When docker:stable (alpine) has python 3.9...
    #  return [x.removesuffix("_component_version") for x in cmd().splitlines()]
    return [
        x[3 : -len("_VERSION")].lower()
        for x in cmd().splitlines()
        if "_VERSION" in x and not "CUDA_LIB" in x
    ]


def load_rc_push_repos_manifest_yaml():
    push_repo_path = pathlib.Path("manifests/rc-push-repos.yml")
    log.debug(f"Loading push repos manifest: {push_repo_path}")
    obj = {}
    with open(push_repo_path, "r") as f:
        obj = yaml.load(f, yaml.Loader)
    return obj




def supported_distro_list_by_cuda_version(
    manifest: dict[str, dict[str, str]], version: str
) -> List[str]:
    """ """
    keys = manifest[f"cuda_v{version}"].keys()
    theset = set()
    for key in keys:
        for f in supported_platforms.list:
            if f.full_name() in key:
                theset.add(f.full_name())
    return list(sorted(theset))


def supported_arch_list(
    manifest: dict[str, dict[str, str]], distro: str, cuda_version: str
) -> List[str]:
    ls = []
    for k in glom.glom(
        manifest,
        glom.Path(f"cuda_v{cuda_version}", distro),
    ):
        if k in supported_platforms.all_architectures_by_common_name():
            ls.append(k)
    return ls
