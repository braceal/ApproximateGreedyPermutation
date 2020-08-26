from setuptools import setup, find_packages
import versioneer
import os


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements", 'install.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))


setup(
    name='DiverseSMILES',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='https://github.com/braceal/DiverseSMILES',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.6",
    author="braceal",
    maintainer='Alex Brace',
    maintainer_email='braceal@umich.edu',
    license='MIT',
)
