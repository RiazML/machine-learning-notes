import os
import re
import sys

def audit_repository():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, "docs")
    files = sorted([f for f in os.listdir(docs_path) if re.match(r"^\d{3}_.*\.md$", f)])
    
    print(f"Auditing {len(files)} study guide markdown files...")
    
    placeholders = ["TODO", "placeholder", "insert here", "empty summary", "write content", "tbd", "to be filled"]
    placeholders_regex = re.compile(r"\b(" + "|".join(placeholders) + r")\b", re.IGNORECASE)
    
    errors = []
    
    for file_name in files:
        file_path = os.path.join(docs_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Word count validation
        word_count = len(content.split())
        if word_count < 100 or len(content.strip()) == 0:
            errors.append(f"{file_name}: Content is too short ({word_count} words).")
            continue
            
        # Placeholder check
        matches = placeholders_regex.findall(content)
        if matches:
            errors.append(f"{file_name}: Contains placeholder words: {set(matches)}")
            
        # Python syntax check
        python_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
        for idx, block in enumerate(python_blocks):
            try:
                compile(block, f"{file_name}_block_{idx}", "exec")
            except SyntaxError as e:
                errors.append(f"{file_name} [Python block {idx}]: SyntaxError: {e.msg} on line {e.lineno}")
                
        # Mermaid validation
        mermaid_blocks = re.findall(r"```mermaid\n(.*?)\n```", content, re.DOTALL)
        for idx, block in enumerate(mermaid_blocks):
            # Check for non-standard arrows
            if any(char in block for char in ["─", "►", "▼", "▲", "◄"]):
                errors.append(f"{file_name} [Mermaid block {idx}]: Contains non-standard arrow symbols (e.g. ─, ►, etc.). Use standard arrows like -->, ---, -.-, or ->>.")
                
            # Check for unquoted special characters in label blocks e.g. Reg(Regression (Continuous))
            # Match flowchart node labels like ID(text), ID[text], ID{text}, ID((text))
            # Focus on parentheses, colons, arithmetic inside labels that are not double-quoted
            lines = block.split("\n")
            for line_no, line in enumerate(lines, 1):
                # Search for unquoted text containing parentheses or special characters inside shape containers
                # Match pattern like: node_id[label containing () or : or arithmetic]
                # If there's double quotes inside, it's fine.
                # A common error is: Reg[Regression (Continuous)] instead of Reg["Regression (Continuous)"]
                for shape_open, shape_close in [("[", "]"), ("(", ")"), ("{", "}"), ("((", "))"), ("[\"", "\"]"), ("(\"", "\")")]:
                    if shape_open.endswith('"'):
                        continue # Already quoted
                    
                    # Look for something like: ID[Text] or ID(Text)
                    # We regex search for a word followed by open shape, then contents not starting with a quote, then close shape
                    # For example: r'\b\w+\[' + r'([^"].*?)' + r'\]'
                    escaped_open = re.escape(shape_open)
                    escaped_close = re.escape(shape_close)
                    pattern = r'\b\w+' + escaped_open + r'([^"].*?)' + escaped_close
                    for label_match in re.findall(pattern, line):
                        # If the label match has special characters like parentheses, colons, or math symbols and isn't quoted
                        if any(c in label_match for c in ["(", ")", ":", "+", "-", "*", "/", "=", "&"]):
                            # Bypass if it is a database shape with double quotes inside like [("something")]
                            if (label_match.startswith('("') and label_match.endswith('")')) or \
                               (label_match.startswith('("') and label_match.endswith('")')):
                                continue
                            errors.append(f"{file_name} [Mermaid line {line_no}]: Unquoted special characters in label '{label_match}'. Please wrap label in double quotes: ID[\"{label_match}\"].")

    print("\n=== Audit Report ===")
    if not errors:
        print("✅ SUCCESS: All checks passed! No placeholders, all Python blocks are syntactically valid, and Mermaid files look correct.")
        return True
    else:
        print(f"❌ FAILURE: Found {len(errors)} issues:")
        for err in errors:
            print(f"  - {err}")
        return False

if __name__ == "__main__":
    success = audit_repository()
    sys.exit(0 if success else 1)
