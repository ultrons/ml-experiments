import util
import random


plain_text = "the story runs parallel to the first game"
plain_int_seq = util.toIntSeq(plain_text)
random.shuffle(plain_int_seq)
cipher_seq = util.toStrSeq(plain_int_seq)
print(cipher_seq)

