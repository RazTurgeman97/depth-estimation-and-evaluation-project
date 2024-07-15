import pathlib

from manager import Manager


def test_ubuntu2204_cuda_keyring_customer_images():
    """Ensures cuda-keyring is used for customer images."""
    _, rc = Manager.run(
        [
            "prog",
            "--manifest=manifests/cuda.yaml",
            "generate",
            "--os-name=ubuntu",
            "--os-version=22.04",
            "--release-label=11.7.0",
            "--cuda-version=11.7.0",
        ],
        exit=False,
    )
    assert rc == 0
    found = False
    with open(pathlib.Path("dist/11.7.0/ubuntu2204/base/Dockerfile")) as file:
        lines = file.readlines()
        for line in lines:
            if "cuda-keyring" in line:
                found = True
    assert found


def test_ubuntu2204_cuda_keyring_rc_images():
    """Ensures cuda-keyring is used for customer images."""
    _, rc = Manager.run(
        [
            "prog",
            "--shipit-uuid=4AFED4F2-D09E-11EC-BCC2-1E8C743E8BA1",
            "generate",
            "--os-name=ubuntu",
            "--os-version=22.04",
            "--release-label=11.8.0",
            "--cuda-version=11.8.0",
        ],
        exit=False,
    )
    assert rc == 0
    found = False
    with open(pathlib.Path("kitpick/ubuntu2204/base/Dockerfile")) as file:
        lines = file.readlines()
        for line in lines:
            if "apt-key" in line:
                found = True
    assert found
