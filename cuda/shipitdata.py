from config import *
from dotdict import *
from retry import retry
from error import *
import utils

import requests
import pathlib
import re
import sys
import json
import yaml

from plumbum.cmd import rm  # type: ignore

from beeprint import pp  # type: ignore


class ShipitInvalidDataError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Error: %s" % self.value


class ShipitData:
    """Class to wrap shipit data."""

    data = DotDict()

    shipit_uuid = ""
    shipit_manifest = DotDict()

    product_name = ""
    candidate_number = 0
    release_label = ""
    distros = None

    l4t_base_image = ""

    tegra: bool = False

    push_repo_logged_in = ""

    def __init__(self, shipit_uuid="", shipit_json=""):
        self.shipit_uuid = shipit_uuid
        if shipit_uuid:
            self.data = DotDict(self.get_shipit_global_json())
        elif shipit_json:
            data = json.loads(shipit_json)
            self.data = DotDict(data)
        if not self.data:
            raise ShipitInvalidDataError(
                "Attempt to initialize ShipitData class without data!"
            )

    def get_shipit_funnel_json(self, distro, distro_version, arch):
        funnel_distro = distro
        if (
            foo := supported_platforms.by_distro(funnel_distro)
        ) and "rpm" in foo.package_format:
            funnel_distro = "rhel"
        self.arch = arch
        modified_distro_version = distro_version.replace(".", "")
        modified_arch = self.arch.replace("_", "-")
        if modified_arch == "arm64":
            log.debug(f"Converting arch '{arch}' into 'sbsa' for container images")
            modified_arch = "sbsa"
        shipit_distro = f"{funnel_distro}{modified_distro_version}"
        if self.tegra:
            shipit_distro = "ubuntu2204"
            modified_arch = arches.tegra.arch

        platform_name = (
            f"{shipit_distro}-{self.data.product_name}-linux-{modified_arch}.json"
        )
        shipit_json = (
            f"https://kitmaker-web.nvidia.com/funnel/{self.shipit_uuid}/{platform_name}"
        )
        log.info(f"Retrieving funnel json from: {shipit_json}")
        return self.get_http_json(shipit_json)

    def get_shipit_global_json(self):
        global_json = (
            f"https://kitmaker-web.nvidia.com/funnel/{self.shipit_uuid}/global.json"
        )
        log.info(f"Retrieving global json from: {global_json}")
        ldata = self.get_http_json(global_json)
        if ldata:
            return ldata

    # Returns a unmarshalled json object
    @retry(
        (RequestsRetry),
        tries=HTTP_RETRY_ATTEMPTS,
        delay=HTTP_RETRY_WAIT_SECS,
        logger=log,
    )
    def get_http_json(self, url):
        r = requests.get(url)
        log.debug("response status code %s", r.status_code)
        #  log.debug("response body %s", r.json())
        if r.status_code == 200:
            log.info("http json get successful")
        else:
            raise RequestsRetry()
        return r.json()

    def pkg_rel_from_package_name(self, name, version):
        log.debug(f"have name: {name} version: {version}")
        # HACK: see above.
        if "cuda-nsight-compute" in name:
            return "1"
        rgx = re.search(rf"[\w\d-]*{version}-(\d)_?", name)
        if rgx:
            log.debug(f"found match: {rgx.group(1)}")
            return rgx.group(1)
        log.debug("Could not match pkgrel from package name!")

    def shipit_components(self, shipit_json, packages):
        components = {}

        fragments = shipit_json["fragments"]

        def fragment_by_name(name):
            name_with_hyphens = name.replace("_", "-")

            # HACK: inconsistencies kill me..
            # cuda-nsight-compute is not in the shipit data so we have to handle it here.
            if "cuda_nsight_compute" in name and not self.tegra:
                return {'name': name_with_hyphens, 'version': self.release_label}

            for _, v in fragments.items():
                for _, v2 in v.items():
                    if any(x in v2["name"] for x in [name, name_with_hyphens]):
                        return v2

        for pkg in packages:
            log.debug(f"package: {pkg}")
            fragment = fragment_by_name(pkg)
            if not fragment:
                log.warning(f"{pkg} was not found in the fragments json!")
                continue

            name = fragment["name"]
            version = fragment["version"]

            pkg_rel = self.pkg_rel_from_package_name(name, version)
            if not pkg_rel:
                raise Exception(
                    (
                        "Could not get package release version from package name "
                        f"'{name}' using version '{version}'. Perhaps there is an issue in the RC data?"
                    )
                )

            pkg_no_prefix = pkg[len("cuda_") :] if pkg.startswith("cuda_") else pkg

            # rename "devel" to "dev" to keep things consistant with ubuntu
            if "_devel" in pkg_no_prefix:
                pkg_no_prefix = pkg_no_prefix.replace("_devel", "_dev")

            log.debug(f"component: {pkg_no_prefix} version: {version} pkg_rel: {pkg_rel}")

            components.update({f"{pkg_no_prefix}": {"version": f"{version}-{pkg_rel}"}})

        return components

    def supported_distros(self):
        """Returns a set of supported distro names for a shipit manifest."""
        distros = set()
        if self.tegra:
            log.debug("TEGRA DETECTED")
            distros.add("ubuntu2004")
            return distros
        for platform in self.data.targets.items():
            for os in platform[1]:
                if supported_platforms.is_supported(os):
                    distros.update(supported_platforms.translated_name(os))
        return distros

    def supported_distros_by_arch(self, arch: str):
        sdistros = set()
        larch = arch
        if "ppc64el" in arch:
            log.debug(f"Setting key '{arch}' to 'ppc64le' for images")
            larch = "ppc64le"
        elif "arm64" in arch:
            log.debug(f"Setting key '{arch}' to 'sbsa' for images")
            larch = "sbsa"
        if not f"linux-{larch}" in self.data.targets:
            log.debug(f"'linux-{larch}' not found in shipit data!")
            return
        for distro in self.data.targets[f"linux-{larch}"]:
            if supported_platforms.is_supported(distro):
                sdistros.update(supported_platforms.translated_name(distro))
        return sdistros

    def kitpick_repo_url(self):
        repo_distro = self.distro
        if (
            foo := supported_platforms.by_distro(repo_distro)
        ) and "rpm" in foo.package_format:
            repo_distro = "rhel"
        clean_distro = "{}{}".format(repo_distro, self.distro_version.replace(".", ""))
        return (
            f"https://kitmaker-web.nvidia.com/"
            f"kitpicks/{self.product_name}/{self.release_label}/{self.candidate_number}/repos/{clean_distro}"
        )

    def generate_shipit_manifest(self, output_path, cudnn_json_path=None):
        # 22:31 Tue Jul 13 2021 FIXME: (jesusa) this function is way too long to be considered good practice in python
        log.debug("Building the shipit manifest")

        self.product_name = self.data["product_name"]
        self.tegra = "tegra" in self.product_name
        self.candidate_number = self.data["cand_number"]
        self.release_label = self.data["rel_label"]
        self.distros = self.supported_distros()

        log.info(f"Product Name: '{self.product_name}'")
        log.info(f"Candidate Number: '{self.candidate_number}'")
        log.info(f"Release label: {self.release_label}")

        reldata = {}
        for plat, distros in self.data["targets"].items():
            if "windows" in plat:
                continue
            log.debug(f"platform: {plat}")
            splat = plat.split("-")
            if len(splat) > 2 or len(splat) <= 1:
                # At the moment, platforms needed by cuda image only contain a name and arch separated by a single hyphen
                continue
            os, arch = splat[0], splat[1]
            if any(a in arch for a in ["sbsa"]):
                log.debug(f"Converting arch '{arch}' into 'arm64' for container images")
                arch = "arm64"
            if not os in reldata:
                reldata[os] = {}
            if not arch in reldata[os]:
                reldata[os][arch] = {}
            reldata[os][arch]["distros"] = self.supported_distros_by_arch(arch)

        if output_path.exists:
            log.warning(f"Removing path '{output_path}'")
            rm["-rf", output_path]()

        output_path.mkdir(parents=True, exist_ok=False)
        self.output_manifest_path = pathlib.Path(f"{output_path}/manifest.yml")

        release_key = f"cuda_v{self.release_label}"
        self.shipit_manifest = {
            release_key: {
                "dist_base_path": output_path.as_posix(),
                "push_repos": ["artifactory"],
            }
        }

        #
        # FIXME: find a better way to do this!
        #
        #        Move this function to the parent class!
        #
        def nested_keys(data):
            for k, v in data.items():
                if isinstance(v, dict):
                    for pair in nested_keys(v):
                        yield (k, *pair)
                else:
                    if "distros" in k:
                        yield ([v])
                    else:
                        yield (k, v)

        manifest = DotDict()
        for platform in nested_keys(reldata):
            os = platform[0]
            self.arch = platform[1]
            if self.tegra:
                self.arch = arches.tegra.common_name
            distros = platform[2]
            if self.tegra and ("aarch64" not in platform):
                log.warning(
                    (
                        f"Skipping platform! '{os}-{self.arch}' "
                        "is not supported for L4T Cuda Container Images (yet)"
                    )
                )
                continue
            log.info(f"Inspecting global.json platform: {platform}")
            for distro in distros:
                if self.tegra and not "ubuntu2204" in distro:
                    # filters out the ubuntu distros in the shipitdata
                    continue
                rgx = re.search(r"(\D*)(\d*)", distro)
                if not self.tegra:
                    if rgx:
                        self.distro = rgx.group(1)
                        self.distro_version = rgx.group(2).replace("04", ".04")
                    else:
                        raise UnknownCudaRCDistro(
                            "Distro '{distro}' has an unknown format"
                        )
                    platform = f"{self.distro}{self.distro_version}"
                else:
                    # Now that we are ready to render the template, change the arch if tegra
                    self.arch == "arm64"
                    self.distro = "ubuntu2204"
                    self.distro_version = ""
                    platform = "l4t"
                    if not cudnn_json_path:
                        log.error("Argument `--cudnn-json-path` is not set!")
                        sys.exit(1)
                    platform = f"{platform}"

                self.output_path = pathlib.Path(f"{output_path}/{platform}")

                sjson = self.get_shipit_funnel_json(
                    self.distro, self.distro_version, self.arch
                )

                pkgs = utils.template_packages(self.distro)

                log.debug(f"template packages: {pkgs}")
                components = self.shipit_components(sjson, pkgs)

                # TEMP WAR: populate cudnn component for L4T
                if self.tegra:
                    cudnn_comp = {
                        "cudnn8": {
                            "version": "",
                            "source": "",
                            "dev": {"source": "", "md5sum": ""},
                        }
                    }
                    log.debug(f"cudnn_json_path: {cudnn_json_path}")
                    with open(pathlib.Path(cudnn_json_path), "r") as f:
                        cudnn = json.loads(f.read())
                    for rawObj in cudnn:
                        artf = DotDict(rawObj)
                        log.debug(f"cudnn json loop item\n{pp(artf, output=False)}")
                        artdir = pathlib.Path(artf.path)
                        artpath = f"https://urm.nvidia.com/artifactory/{artdir}"
                        name = artdir.name
                        if "arm64" in name:
                            if "-dev_" in name:
                                cudnn_comp["cudnn8"]["dev"]["source"] = artpath
                                cudnn_comp["cudnn8"]["dev"]["md5sum"] = artf.md5
                            else:
                                cudnn_comp["cudnn8"]["version"] = artf.props.version[0]
                                cudnn_comp["cudnn8"]["source"] = artpath
                                cudnn_comp["cudnn8"]["md5sum"] = artf.md5
                    if cudnn_comp:
                        log.debug(f"cudnn component\n{pp(cudnn_comp, output=False)}")
                        components.update(cudnn_comp)

                image_name = "gitlab-master.nvidia.com:5005/cuda-installer/cuda/release-candidate/cuda"
                template_path = "templates/ubuntu"
                if not self.tegra and "ubuntu" not in self.distro:
                    template_path = "templates/redhat"
                base_image = f"{self.distro}:{self.distro_version}"
                if "ubi" in self.distro:
                    base_image = (
                        f"registry.access.redhat.com/ubi{self.distro_version}/ubi:latest"
                    )
                requires = ""

                key = "push_repos"
                if self.tegra:
                    key = "l4t_push_repos"
                prepos = utils.load_rc_push_repos_manifest_yaml()[key]
                if not self.push_repo_logged_in:
                    # only need to do this once
                    utils.auth_registries(prepos)
                    self.push_repo_logged_in = True

                if self.tegra:
                    log.debug(f"Show me the object!!\n{pp(self, output=False)}")
                    # Ugh... hardcoded values suck! This will be changed in the future.
                    if self.release_label == "10.2.460":
                        self.l4t_base_image = f"{L4T_BASE_IMAGE_NAME}:r32.7.1"
                        requires = "cuda>=10.2"
                    elif self.release_label == "11.4.14":
                        self.l4t_base_image = f"{L4T_BASE_IMAGE_NAME}:r35.1.0"
                        requires = "cuda>=11.4"
                    elif self.release_label == "11.4.19":
                        self.l4t_base_image = f"{L4T_BASE_IMAGE_NAME}:r35.4.1"
                        requires = "cuda>=11.4"
                    elif self.release_label == "12.2.12":
                        self.l4t_base_image = f"{L4T_BASE_IMAGE_NAME}:ubuntu22.04"
                        requires = "cuda>=12.2"
                    elif not self.l4t_base_image:
                        raise Exception("L4T Base tag not set!")
                    base_image = self.l4t_base_image
                    image_name = (
                        f"gitlab-master.nvidia.com:5005/cuda-installer/cuda/l4t-cuda"
                    )

                self.shipit_manifest["push_repos"] = prepos
                manifest = self.shipit_manifest[release_key]
                manifest["image_name"] = image_name
                if not platform in manifest:
                    manifest[platform] = {
                        "base_image": base_image,
                        "image_tag_suffix": f"-{self.data['cand_number']}",
                        "template_path": template_path,
                        "repo_url": self.kitpick_repo_url(),
                    }

                manifest[platform][f"{self.arch}"] = {
                    "requires": requires,
                    "components": components,
                }

        if not manifest:
            log.warning(
                (
                    "No manifest to write after parsing Shipit data. "
                    "Unsupported kitpick as it doesn't contain anything useful for Cuda Image!"
                )
            )
            sys.exit(155)

        self.shipit_manifest[release_key] = manifest
        log.info(f"Writing shipit manifest: {self.output_manifest_path}")
        self.generate_shipit_manifest_from_manifest(self.shipit_manifest)

    def generate_shipit_manifest_from_manifest(self, manifest):
        yaml_str = yaml.dump(manifest)
        with open(self.output_manifest_path, "w") as f:
            f.write(yaml_str)
