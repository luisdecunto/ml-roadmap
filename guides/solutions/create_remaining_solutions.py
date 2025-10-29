#!/usr/bin/env python3
import os

# Base directory
base_dir = os.getcwd()

# HTML template function
def create_html(title, md_filename, exercise_filename):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - ML Roadmap</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        .markdown-body {{
            line-height: 1.6;
        }}
        .markdown-body h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            color: #1f2937;
            border-bottom: 3px solid #f59e0b;
            padding-bottom: 0.5rem;
        }}
        .markdown-body h2 {{
            font-size: 1.875rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 0.75rem;
            color: #374151;
        }}
        .markdown-body h3 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            color: #4b5563;
        }}
        .markdown-body p {{
            margin-bottom: 1rem;
            color: #374151;
        }}
        .markdown-body ul, .markdown-body ol {{
            margin-bottom: 1rem;
            margin-left: 1.5rem;
        }}
        .markdown-body li {{
            margin-bottom: 0.5rem;
            color: #374151;
        }}
        .markdown-body code {{
            background-color: #f3f4f6;
            padding: 0.125rem 0.375rem;
            border-radius: 0.25rem;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            color: #dc2626;
        }}
        .markdown-body pre {{
            background-color: #1f2937;
            color: #f3f4f6;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin-bottom: 1rem;
        }}
        .markdown-body pre code {{
            background-color: transparent;
            padding: 0;
            color: #f3f4f6;
        }}
        .markdown-body a {{
            color: #2563eb;
            text-decoration: underline;
        }}
        .markdown-body a:hover {{
            color: #1d4ed8;
        }}
        .markdown-body hr {{
            margin: 2rem 0;
            border: 0;
            border-top: 2px solid #e5e7eb;
        }}
        .markdown-body blockquote {{
            border-left: 4px solid #f59e0b;
            padding-left: 1rem;
            color: #6b7280;
            font-style: italic;
            margin: 1rem 0;
        }}
        .markdown-body strong {{
            color: #1f2937;
            font-weight: 700;
        }}
    </style>
</head>
<body class="bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 min-h-screen">
    <div class="max-w-4xl mx-auto p-4 md:p-8">
        <!-- Header -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div class="flex items-center justify-between mb-4">
                <h1 class="text-2xl font-bold text-gray-800">{title}</h1>
                <div class="flex space-x-2">
                    <a href="../exercises/{exercise_filename}" class="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-lg text-sm font-semibold">
                        Back to Exercises
                    </a>
                    <a href="../../index.html" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-semibold">
                        ← Roadmap
                    </a>
                </div>
            </div>
            <div class="bg-yellow-50 border border-yellow-300 rounded-lg p-4">
                <p class="text-sm text-yellow-800">
                    <strong>⚠️ Try the exercises first!</strong> These solutions are here to help you check your work and learn from mistakes. Working through problems yourself is crucial for building intuition.
                </p>
            </div>
        </div>

        <!-- Content -->
        <div class="bg-white rounded-lg shadow-lg p-8">
            <div id="content" class="markdown-body"></div>
        </div>

        <!-- Footer -->
        <div class="mt-6 text-center text-gray-600 text-sm space-x-4">
            <a href="../exercises/{exercise_filename}" class="text-amber-600 hover:text-amber-700 font-semibold">
                ← Back to Exercises
            </a>
            <span>|</span>
            <a href="../../index.html" class="text-indigo-600 hover:text-indigo-700 font-semibold">
                Back to ML Roadmap
            </a>
        </div>
    </div>

    <script>
        // Read markdown content from the .md file
        fetch('{md_filename}')
            .then(response => response.text())
            .then(markdown => {{
                document.getElementById('content').innerHTML = marked.parse(markdown);
            }})
            .catch(error => {{
                console.error('Error loading markdown:', error);
                document.getElementById('content').innerHTML = '<p class="text-red-600">Error loading content. Please check that {md_filename} exists.</p>';
            }});
    </script>
</body>
</html>
"""

# Create HTML files
html_configs = [
    ("Information Theory Solutions", "information_theory_solutions.md", "information_theory_exercises.html"),
    ("Neural Networks Solutions", "neural_networks_solutions.md", "neural_networks_exercises.html"),
    ("CNN Solutions", "cnn_solutions.md", "cnn_exercises.html"),
    ("Transformer Solutions", "transformer_solutions.md", "transformer_exercises.html"),
]

for title, md_file, ex_file in html_configs:
    html_filename = md_file.replace('.md', '.html')
    html_content = create_html(title, md_file, ex_file)
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Created {html_filename}")

print("\nAll HTML files created successfully!")
