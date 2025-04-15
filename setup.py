from setuptools import setup, find_packages

setup(
    name='feature_viewer',         # Replace with your desired package name
    version='0.0.0',                 # Version of your package
    packages=find_packages(),        # Automatically find packages in your directory
    install_requires=[               # List any dependencies here (e.g., 'numpy', 'requests')
        # 'dependency_one>=1.0.0',
    ],
    author='Butian Xiong',
    author_email='xiongbutian768@gmail.com',
    description='A test',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-repo',  # Update with your repository URL if available
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)