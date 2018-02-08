from setuptools import setup, find_packages

setup(
    name = 'visual_coding_2p_analysis',
    version = '0.1.0',
    description = """Visual Coding 2P analysis code""",
    author = "Nicholas Cain",
    author_email = "nicholasc@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/visual_coding_2p_analysis',
    packages = find_packages(),
    include_package_data=True,
    entry_points={
          'console_scripts': [
              'visual_coding_2p_analysis = visual_coding_2p_analysis.__main__:main'
        ]
    },
    setup_requires=['pytest-runner'],
)
