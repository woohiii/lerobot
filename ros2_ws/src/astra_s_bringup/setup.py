from glob import glob

from setuptools import setup

package_name = "astra_s_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/params", glob("params/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="youngchan",
    maintainer_email="fbwiei14@gmail.com",
    description="Astra S launch/params wrapper around the astra_camera driver",
    license="Apache-2.0",
)
