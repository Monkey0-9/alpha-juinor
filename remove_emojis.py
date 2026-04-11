import os
import re

replacements = {
    '': '',
    '': '',
    '[INIT]': '[INIT]',
    '[SYS]': '[SYS]',
    '[SYS]': '[SYS]',
    '[STAT]': '[STAT]',
    '[CHK]': '[CHK]',
    '[ALRT]': '[ALRT]',
    '[WARN]': '[WARN]',
    '[WARN]': '[WARN]',
    '[HALT]': '[HALT]',
    '[HALT]': '[HALT]',
    '[X]': '[X]',
    '[O]': '[O]',
    '[TIME]': '[TIME]',
    '[OBS]': '[OBS]',
    '[TIME]': '[TIME]',
    '[NET]': '[NET]',
    '[UP]': '[UP]',
    '[TGT]': '[TGT]',
    '[LST]': '[LST]',
    '[BUY]': '[BUY]',
    '[SEL]': '[SEL]',
    ' ': ' ',
    '': ''
}

for root, dirs, files in os.walk('.', topdown=True):
    # skip .venv, .git, etc
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for str_file in files:
        if str_file.endswith('.py'):
            path = os.path.join(root, str_file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content
                for k, v in replacements.items():
                    content = content.replace(k, v)

                if content != original_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Updated {path}")
            except Exception as e:
                print(f"Skipping {path}: {e}")
