# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = []
binaries += collect_dynamic_libs('lightgbm')


a = Analysis(
    ['e:\\Project\\python\\KoloVirusDetector_ML_V2-main\\scanner_service.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=['lightgbm.basic', 'sklearn.preprocessing'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'seaborn', 'pandas', 'optuna', 'hyperopt', 'fast_hdbscan', 'tqdm', 'torch'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Axon_ml',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Axon_ml',
)
