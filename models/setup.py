from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("gpt_request.py", compiler_directives={"language_level": "3"}),
)