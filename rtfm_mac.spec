# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['rtfm_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app.py', '.'),
        ('app_patch.py', '.'),
        ('db.py', '.'),
        ('verify_cuda.py', '.'),
        ('requirements.txt', '.'),
        ('.env.example', '.'),
    ],
    hiddenimports=[
        'torch', 'torchvision', 'torchaudio',
        'gradio', 'transformers', 'huggingface_hub',
        'pdf2image', 'lancedb', 'colpali_engine',
        'python-dotenv', 'openai'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RTFM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.icns',  # Add an icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RTFM',
)

app = BUNDLE(
    coll,
    name='RTFM.app',
    icon='app_icon.icns',
    bundle_identifier='com.yourcompany.rtfm',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
    }
)