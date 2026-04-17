import os

def clean_content(content):
    # Use hex or escape sequences to avoid the script cleaning itself
    HLD = '[' + 'HLD' + ']'
    CLK = '[' + 'CLK' + ']'
    OK = '[' + 'OK' + ']'
    ERR = '[' + 'ERR' + ']'
    
    content = content.replace(HLD, ' ')
    content = content.replace(CLK, '')
    content = content.replace(OK, '')
    content = content.replace(ERR, '')
    return content

def find_and_clean():
    HLD = '[' + 'HLD' + ']'
    CLK = '[' + 'CLK' + ']'
    
    for root, dirs, files in os.walk('.'):
        # Skip directories that might contain large amounts of data or irrelevant files
        if any(x in root for x in ['qc-env', '.git', '__pycache__', '.mypy_cache', 'data\\cache']):
            continue
            
        for file in files:
            if file.endswith(('.py', '.sh', '.bat', '.ps1', '.txt', '.md', '.yaml', '.yml', '.json', '.toml')):
                path = os.path.join(root, file)
                if path.endswith('clean_corruption.py'):
                    continue
                    
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if HLD in content or CLK in content:
                        print(f"Cleaning {path}")
                        cleaned = clean_content(content)
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(cleaned)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    find_and_clean()
