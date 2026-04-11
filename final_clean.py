import os

def clean_file(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        
        # Define markers using hex/concatenation to avoid self-cleaning
        CLK = b'[' + b'CLK' + b']'
        HLD = b'[' + b'HLD' + b']'
        OK = b'[' + b'OK' + b']'
        ERR = b'[' + b'ERR' + b']'
        
        if CLK in data or HLD in data:
            print(f"Cleaning: {path}")
            cleaned = data.replace(HLD, b' ')
            cleaned = cleaned.replace(CLK, b'')
            cleaned = cleaned.replace(OK, b'')
            cleaned = cleaned.replace(ERR, b'')
            
            with open(path, 'wb') as f:
                f.write(cleaned)
            return True
    except Exception:
        pass
    return False

def main():
    count = 0
    skip_dirs = {'.git', 'qc-env', 'data/cache', '__pycache__', '.mypy_cache', '.pytest_cache', '.venv', '.trunk', 'mlruns'}
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(('.py', '.md', '.yaml', '.json', '.txt', '.sh', '.bat', '.ps1')):
                path = os.path.join(root, file)
                if clean_file(path):
                    count += 1
    print(f"Total files cleaned: {count}")

if __name__ == "__main__":
    main()
