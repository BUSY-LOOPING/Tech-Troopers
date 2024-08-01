from setuptools import setup, find_packages

setup(
    name='drone_troopers',
    version='0.0.1',
    author='Tech Troopers',
    author_email='dhruvyadav2905@gmail.com',
    description='A module for drone data processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url='https://github.com/yourusername/drone_troopers',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'xgboost',
        # Example: 'numpy>=1.18.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
