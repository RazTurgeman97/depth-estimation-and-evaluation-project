import logging

from dataclasses import dataclass, field
from typing import List, Optional
from collections import namedtuple

from dotdict import DotDict

# from beeprint import pp

log = logging.getLogger()

HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_WAIT_SECS = 30

SUPPORTED_DISTRO_LIST = ["ubuntu", "ubi", "centos", "rockylinux"]
L4T_BASE_IMAGE_NAME = "nvcr.io/nvidian/nvidia-aarch64-base"

# Increased buildkit version from 0.8.1 to 0.10.3 to overcome ubuntu 22.04 build failures
MOBY_BUILDKIT_VERSION = "v0.10.3"


@dataclass(frozen=True)
class SupportedArchitecture:
    """ """

    arch: str
    common_name: str
    repo_name: tuple
    nv_name: str = ""
    container_arch: str = ""


@dataclass(frozen=True)
class SupportedPlatform:
    """ """

    distro: str
    version: str
    arches: List[SupportedArchitecture]
    package_format: str
    image_name: str = ""
    flavor: str = ""

    def full_name(self) -> str:
        """ """
        return f"{self.distro}{self.version}"

    def shipit_distro_name(self) -> str:
        """ """
        return f"{self.distro}{self.version.replace('.','')}"

    def shipit_distro_arch(self, shipit_arch_key: str) -> bool:
        """ """
        return any(x.arch in shipit_arch_key for x in self.arches)

    def rhel_based_supported(self, distro: str) -> bool:
        return (
            "rhel" in distro and "rpm" in self.package_format and self.version in distro
        )

    def common_arches_csv(self) -> str:
        """ """
        csv: List[str] = []
        for arch in self.arches:
            csv.append(arch.common_name)
        return ", ".join(csv)

    def container_arches_csv(self) -> str:
        """ """
        csv: List[str] = []
        for arch in self.arches:
            csv.append(arch.container_arch)
        return ", ".join(csv)


_package_repo_arch_repr = namedtuple("_package_repo_arch_repr", ["deb", "rpm"])

arches = DotDict(
    {
        "arm64": SupportedArchitecture(
            arch="arm64",
            common_name="arm64",
            repo_name=_package_repo_arch_repr(deb="sbsa", rpm="sbsa"),
            container_arch="arm64",
            nv_name="sbsa",
        ),
        "x86_64": SupportedArchitecture(
            arch="x86_64",
            common_name="x86_64",
            repo_name=_package_repo_arch_repr(deb="x86_64", rpm="x86_64"),
            container_arch="x86_64",
            nv_name="x86_64",
        ),
        "tegra": SupportedArchitecture(
            arch="aarch64",
            common_name="tegra",
            repo_name=_package_repo_arch_repr(deb="arm64", rpm="arm64"),
            container_arch="arm64",
            nv_name="tegra",
        ),
    }
)


def _pop_supported_platforms():
    return [
        SupportedPlatform(
            distro="ubuntu",
            version="22.04",
            arches=[arches.arm64, arches.x86_64],
            package_format="deb",
        ),
        SupportedPlatform(
            distro="ubuntu",
            version="20.04",
            arches=[arches.arm64, arches.x86_64],
            package_format="deb",
        ),
        SupportedPlatform(
            distro="ubi",
            version="9",
            # ubi9 supports ppc64le, but CUDA does not...
            arches=[arches.x86_64, arches.arm64],
            package_format="rpm",
        ),
        SupportedPlatform(
            distro="rockylinux",
            version="9",
            # rockylinux9 supports ppc64le, but CUDA does not...
            arches=[arches.x86_64, arches.arm64],
            package_format="rpm",
        ),
        SupportedPlatform(
            distro="ubi",
            version="8",
            arches=[arches.x86_64, arches.arm64],
            package_format="rpm",
        ),
        SupportedPlatform(
            distro="rockylinux",
            version="8",
            arches=[arches.x86_64, arches.arm64],
            package_format="rpm",
        ),
        SupportedPlatform(
            distro="l4t",
            version="",
            arches=[arches.tegra],
            package_format="deb",
            flavor="jetson",
            image_name="gitlab-master.nvidia.com:5005/cuda-installer/cuda/l4t-cuda",
        ),
    ]


@dataclass(frozen=True)
class SupportedPlatformsList:
    list: List[SupportedPlatform] = field(default_factory=_pop_supported_platforms)

    def is_supported(self, distro: str) -> bool:
        """ """
        for p in self.list:
            if distro in p.shipit_distro_name() or (
                p.rhel_based_supported(distro) and p.version in distro
            ):
                return True
        return False

    def translated_name(self, distro: str) -> List[str]:
        """ """
        name: List[str] = []
        for p in self.list:
            if distro in p.shipit_distro_name() or (
                p.rhel_based_supported(distro) and p.version in distro
            ):
                name.append(p.shipit_distro_name())
        assert name
        return name

    # def by_distro(self, distro: str) -> TSupportedPlatform:
    def by_distro(self, distro: str) -> Optional[SupportedPlatform]:
        """ """
        for x in self.list:
            if distro in x.shipit_distro_name():
                return x
        return None

    def all_architectures_by_nvidia_name(self) -> List[str]:
        """ """
        theset = set()
        for x in self.list:
            for y in x.arches:
                theset.add(y.nv_name)
        return list(sorted(theset))

    def all_architectures_by_common_name(self) -> List[str]:
        """ """
        theset = set()
        for x in self.list:
            for y in x.arches:
                theset.add(y.common_name)
        return list(sorted(theset))


supported_platforms = SupportedPlatformsList()
