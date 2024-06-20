from setuptools import setup
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        print('Hello! Just make sure you have TensorFlow>=2.0 and Matplotlib>=3 installed. Siddique!')
        install.run(self)

setup(
    name='trueloss',
    version='0.1.3',
    author='Siddique Abusaleh',
    author_email='trueloss.py@gmail.com',
    description='A library for computing true training loss in Keras models without regularization effects.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/trueloss/trueloss',
    py_modules=['trueloss'],
    install_requires=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Environment :: Console',
        'Natural Language :: English',
    ],
    python_requires='>=3.9',
    cmdclass={
        'install': CustomInstallCommand,
    },
)
