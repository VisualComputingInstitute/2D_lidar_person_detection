from setuptools import setup, find_packages

setup(
    name="dr_spaam",
    version="1.2.0",
    author="Dan Jia",
    author_email="jia@vision.rwth-aachen.de",
    packages=find_packages(include=["dr_spaam", "dr_spaam.*", "dr_spaam.*.*"]),
    license="LICENSE.txt",
    description="DR-SPAAM, a deep-learning based person detector for 2D range data.",
)
