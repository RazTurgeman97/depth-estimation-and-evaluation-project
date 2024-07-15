from shipitdata import *

from beeprint import pp  # type: ignore


def test_supported_distros():
    """Ensure distros for "arm64" are pulled from sbsa shipit data."""

    input_json = """{
   "cand_number" : "004",
   "product_name" : "cuda-r11-4",
   "rel_label" : "11.4.0",
   "targets" : {
      "linux-aarch64" : [
         "archives",
         "ubuntu2004"
      ],
      "linux-ppc64le" : [
         "archives",
         "conda",
         "rhel8",
         "runfile"
      ],
      "linux-sbsa" : [
         "archives",
         "conda",
         "rhel8",
         "rhel9",
         "runfile",
         "sles15",
         "ubuntu2004",
         "ubuntu2204"
      ],
      "linux-x86_64" : [
         "archives",
         "conda",
         "debian11",
         "fedora36",
         "opensuse15",
         "rhel7",
         "rhel8",
         "rhel9",
         "runfile",
         "sles15",
         "ubuntu1804",
         "ubuntu2004",
         "ubuntu2204",
         "wsl-ubuntu"
      ],
      "windows-x86_64" : [
         "archives",
         "conda",
         "windows"
      ]
   }
}
"""
    sd = ShipitData(shipit_json=input_json)
    pp(sd)
    assert sd.supported_distros() == {
        "ubuntu2204",
        "ubuntu2004",
        "ubuntu1804",
        # "ubi9",
        "ubi8",
        "rockylinux8",
        "ubi7",
        "centos7",
    }


def test_supported_distros_by_arch_arm64():
    """Ensure distros for "arm64" are pulled from sbsa shipit data."""

    input_json = """{
   "cand_number" : "003",
   "product_name" : "cuda-r11-4",
   "rel_label" : "11.4.0",
   "targets" : {
      "linux-aarch64" : [
         "archives",
         "ubuntu2004"
      ],
      "linux-sbsa" : [
         "archives",
         "conda",
         "rhel8",
         "ubuntu2004"
      ],
      "linux-x86_64" : [
         "archives",
         "conda",
         "rhel7",
         "rhel8",
         "ubuntu1804",
         "ubuntu2004"
      ]
   },
   "web_user" : "foo"
}"""
    sd = ShipitData(shipit_json=input_json)
    pp(sd)
    assert sd.supported_distros_by_arch("arm64") == {"ubi8", "rockylinux8", "ubuntu2004"}


def test_supported_distros_by_arch_l4t():
    """Ensure distros for "l4t" are pulled from sbsa shipit data."""

    input_json = """{
   "cand_number" : "009",
   "product_name" : "cuda-r11-4-tegra",
   "rel_label" : "11.4.14",
   "targets" : {
      "eb-x86_64" : [
         "eb"
      ],
      "l4t-aarch64" : [
         "archives"
      ],
      "linux-aarch64" : [
         "archives",
         "l4t",
         "ubuntu1804",
         "ubuntu2004"
      ],
      "linux-x86_64" : [
         "archives",
         "ubuntu1804",
         "ubuntu2004"
      ],
      "qnx-standard-aarch64" : [
         "qnx-standard"
      ]
   },
   "web_user" : "foo"
}"""
    sd = ShipitData(shipit_json=input_json)
    pp(sd)
    # Yes, l4t is an aarch64 arch not arm64, this is not accurate in our repos because everyone outside of Nvidia uses arm64
    # Including container images...
    assert sd.supported_distros_by_arch("aarch64") == {"l4t", "ubuntu1804", "ubuntu2004"}
