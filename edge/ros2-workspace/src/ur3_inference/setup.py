from setuptools import setup

package_name = 'ur3_inference'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Physical AI Toolchain',
    maintainer_email='noreply@example.com',
    description='Real-time TensorRT inference node for the UR3 pick-and-place policy.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Enables: ros2 run ur3_inference inference_node
            'inference_node = ur3_inference.ur3_inference_node:main',
        ],
    },
)
