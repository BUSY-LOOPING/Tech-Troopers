import sys
import subprocess
import os
import tempfile
import shutil


def get_resource_path(filename):
    """Get the absolute path of a resource file."""
    return os.path.join(os.path.dirname(__file__), filename)

def escape_latex_special_chars(filename):
    """Escape special characters in the filename for LaTeX."""
    special_chars = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde',
        '^': '\\textasciicircum',
        '\\': '\\textbackslash'
    }
    for char, escape in special_chars.items():
        filename = filename.replace(char, escape)
    return filename

def update_latex_content(filename):
    # Path to the LaTeX file
    latex_file_path = get_resource_path('main.tex')
    
    # Escape special characters in the filename
    escaped_filename = escape_latex_special_chars(os.path.basename(filename))
    
    # Read the existing content of the LaTeX file
    with open(latex_file_path, 'r') as file:
        content = file.readlines()

    # Replace the line containing the filename
    for i, line in enumerate(content):
        if 'filename.csv' in line:
            content[i] = line.replace('filename.csv', escaped_filename)
            break
        
    return '\n'.join(content)

def compile_raw_latex(raw_latex_code, resource_paths=['img']):
    with tempfile.TemporaryDirectory() as tempdir:
        tex_file_path = os.path.join(tempdir, 'temp.tex')
        
        # Write the raw LaTeX code to a temporary file
        with open(tex_file_path, 'w') as tex_file:
            tex_file.write(raw_latex_code)
            
        for resource_dir in resource_paths:
            resource_dir_path = get_resource_path(resource_dir)
            for item in os.listdir(resource_dir_path):
                s = os.path.join(resource_dir_path, item)
                d = os.path.join(tempdir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            
        subprocess.run(['pdflatex', '-jobname=report', tex_file_path], check=True)
            
    # # Use subprocess to run pdflatex and pass the raw LaTeX code
    # process = subprocess.Popen(
    #     ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-jobname=report', tex_file_path],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True,
    #     cwd=get_resource_path(tempdir)  # Set the working directory
    # )
    
    # # Send the raw LaTeX code to pdflatex
    # stdout, stderr = process.communicate()
    
    # # Check for errors
    # if process.returncode != 0:
    #     print("Error in LaTeX compilation:")
    #     print(stderr)
    # else:
    #     print("LaTeX compilation successful!")
    #     print(stdout)

def compile_latex():
    latex_file_path = get_resource_path('main.tex')
    # Compile the LaTeX file to PDF
    subprocess.run(['pdflatex', '-jobname=report', latex_file_path], check=True)

def open_pdf():
    # Open the generated PDF file
    if os.name == 'nt':  # For Windows
        os.startfile('main.pdf')
    elif os.name == 'posix':  # For macOS and Linux
        subprocess.run(['open', 'report.pdf'])  # For macOS
        # subprocess.run(['xdg-open', 'report.pdf'])  # For Linux

if __name__ == "__main__":
    # Check if a filename is provided as a command-line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        path1 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\50 throttle not enough power(annotated).xlsx'
        filename = path1  # Default filename if none provided

    content = update_latex_content(filename)
    print(f'Updated filename in main.tex to: {filename}')

    compile_raw_latex(content)
    print('Compilation completed.')

    # open_pdf()
    print('Opened report.pdf.')
