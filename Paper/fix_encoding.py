"""Fix encoding corruption and normalize special chars to LaTeX equivalents."""
import sys

path = r'c:\Users\jaime\OneDrive\Cloud Sabana\MIT\optimizing_pallet_configuration\Paper\manuscript.tex'

with open(path, 'rb') as f:
    raw = f.read()

# Attempt 1: read as UTF-8
try:
    text = raw.decode('utf-8')
    # Signature of double-encoding corruption: UTF-8 bytes read as latin-1 then re-encoded
    if 'â€' in text:
        # Decode the corrupted content: latin-1 -> bytes -> utf-8
        text = raw.decode('latin-1').encode('latin-1').decode('utf-8')
        print('Fixed double-encoding corruption')
    else:
        print('UTF-8 encoding OK, no corruption detected')
except UnicodeDecodeError:
    # File is latin-1
    text = raw.decode('latin-1')
    print('File was latin-1, converting')

# Normalize unicode typographic chars to LaTeX
replacements = [
    ('—', '---'),   # em-dash
    ('–', '--'),    # en-dash
    ('’', "'"),     # right single quote
    ('‘', '`'),     # left single quote
    ('“', '``'),    # left double quote
    ('”', "''"),    # right double quote
    ('â', '---'),  # corrupted em-dash (just in case)
]
for old, new in replacements:
    count = text.count(old)
    if count:
        text = text.replace(old, new)
        print(f'Replaced {count}x {repr(old)} -> {repr(new)}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Saved {len(text)} chars to {path}')
