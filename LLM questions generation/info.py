# The different paths to your files
XML_PATH = "/data/xml"
CSV_PATH = "/data/csv"
QUESTIONS_PATH = "/data/questions"
ANNOTATION_SHEETS_PATH = "/data/annotation_sheets"
CORPUS_INFO_PATH = "/data/corpus.csv"

# The models you want to load / use
MODEL_NAMES = ["qwen3.5:27b"]

# Your prompt
# The {{TAG}} allow you to insert dynamic content into the prompt (ie. content you can change when using the prompt).
PROMPT = f"""
The following is a dialogue where each line is in the format <speaker>: <utterance>. All the <special tokens> indicate extra linguistic information, such as pauses, vocalizations, etc. You should interpret them as extra information that can help you understand the dialogue better, but they are not part of the utterance itself.
You are an expert computational linguist specialized in discourse analysis, pragmatics, and conversational dynamics. You analyze conversational structures and anticipate the natural trajectory and flow of dialogues based on sociolinguistic cues and pragmatic aspects of the conversation.

{{conversational_context}}

After reading the previous dialogue, put yourself in the shoes of an active, engaged participant in the dialogue. Formulate 3 salient, progressive questions about a specific aspect of this last utterance"{{anchor_utterance}}" that you are naturally curious about, and which remain unanswered in the provided context.
Generate only the questions, one per line, without any additional text or explanation.

- - - - - - -

To give you more context on the theoretical part behind this:

- “Question Under Discussion” (QUD) plays a central role in constraining interpretation and guiding discourse flow. In this framework, discourse is viewed as a field where interlocutors pursue the common goal of sharing information. The QUD stack represents the ordered set of questions that the interlocutors are committed to answering at a given point in the discourse.
The QUD framework emphasizes retrievability and relevance. For a communicative act to be rational and cooperative, the speaker’s intended meaning must be retrievable by the addressee, a process facilitated by the structure of the discourse and the QUD.
Relevance is strictly defined in terms of the QUD: a move is relevant if it addresses the question currently under discussion. 
"""
