# setuptoolsのfind_packages関数、setup関数を使えるようにする
from setuptools import find_packages, setup
# osパッケージに含まれる関数群を使用可能にする
import os
# globパッケージからglob関数を使用可能にする
from glob import glob

package_name = 'pcd_convert'

setup(
    # パッケージ名を指定
    name=package_name,
    # パッケージのバージョンを指定
    version='0.0.0',
    # pythonのパッケージディレクトリを指定、testはテストコードを入れておくディレクトリなので除外する。
    packages=find_packages(exclude=['test']),
    data_files=[
        # 追加データなどを入れるリソースディレクトリを指定
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xmlというパッケージの依存を管理するファイルをインストール
        ('share/' + package_name, ['package.xml']),
        # Launch関連をインストールするためにリストアップ
        (os.path.join('share', package_name), glob('./launch/*.launch.xml')),
        # During installation, we need to copy the launch files
        (os.path.join('share', package_name, "launch"), glob('launch/*.launch.py')),
        # Same with the RViz configuration file.
        (os.path.join('share', package_name, "config"), glob('config/*')),
    ],
    # setuptoolsを使ってこのパッケージをインストールすることを指定
    install_requires=['setuptools'],
    zip_safe=True,
    # パッケージのメンテナ（動かないときに連絡窓口になるひと）の名前
    maintainer='hosei',
    # メンテナーの連絡先
    maintainer_email='hosei',
    # パッケージの説明
    description='pcd convert for ROS 2.',
    # パッケージのライセンスを指定
    license='Apache-2.0',
    # 単体テストのため依存を追加
    tests_require=['pytest'],
    # ros2 runコマンドやros2 launchコマンドでノードを起動できるようにするための設定。
    # ここを忘れていると実行ができません。
    entry_points={
        'console_scripts': [
            'pcd_rotation = pcd_convert.pcd_rotation:main',
            'pcd_height_segmentation = pcd_convert.pcd_height_segmentation:main',
            'pcd_reflect_segmentation = pcd_convert.pcd_reflect_segmentation:main',
            'pcd_rotation_lidar2 = pcd_convert.pcd_rotation_lidar2:main',
            'pcd_merge = pcd_convert.pcd_merge:main'
        ],
    },
)
