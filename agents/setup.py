from setuptools import setup, find_packages

setup(
    name="contentrunway",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.0.60",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.10",
        "openai>=1.3.7",
        "pydantic>=2.5.0",
    ],
)