import docutils.core

docutils.core.publish_file(
    source_path = "index.rst",
    destination_path = "output.html",
    writer_name = "html"
)
