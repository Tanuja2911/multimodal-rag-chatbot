def summarize_document(context):
    """
    Tool: Summarize document content.
    """
    if not context.strip():
        return "No document content available to summarize."

    return (
        "Here is a concise summary of the document:\n\n"
        + context[:800]
    )
