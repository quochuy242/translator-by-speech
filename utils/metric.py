import jiwer
from utils import const

jiwer_trans = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
