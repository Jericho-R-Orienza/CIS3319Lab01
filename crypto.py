
from copy import deepcopy
import random
from typing import Iterable


class KeyManager:
    @staticmethod
    def read_key(key_file: str) -> bytes:
        with open(key_file, 'rb') as f:
            return f.read()
    
    @staticmethod
    def save_key(key_file: str, key: bytes):
        with open(key_file, 'wb') as f:
            f.write(key)

    def __init__(self, seed=None):
        self.random = random.Random(seed)
    
    def generate_key(self, key_len=256) -> bytes:
        """"
        Generate a random key of length key_len (bit length).
        return: random bytes of length (key_len // 8)
        """
        # TODO: your code here
        #rand_bytes_key = bytes(random.randint(0, 255) for _ in range(num_bytes)) -- this line gets the job done but it is not secure
        #https://stackoverflow.com/questions/47514695/whats-the-difference-between-os-urandom-and-random
        num_bytes = key_len // 8
        rand_bytes = os.urandom(num_bytes) #note: we are outputting binary bytes
        return rand_bytes

#use map() function https://www.geeksforgeeks.org/python-map-function/
def bitize(byts: bytes) -> 'list[int]': #note: parameter: byte type called byts; return type: a list of integers
    """
    bitize bytes
    """
    bits = []
    # TODO: your code here
    for byte in byts:
        #use bin() to convert to binary, and then we use [2:] to skip the "0b" prefix found in all binary -- finally, zfill(8) is our padding to ensure that all binary bytes are converted into 8 bits and nothing less
        binary_string = bin(byte)[2:].zfill(8)
        bits.extend(map(int, binary_string)) #.extend() is used to add elements from an iterable -- which is map(int, binary-string) converting the binary string into a list of ints
    return bits

def debitize(bits: Iterable[int]) -> bytes:
    """
    debbitize a list of bits
    """
    if len(bits) % 8 != 0:
        raise ValueError('bits length is not a multiple of 8')

    byts = []

    # TODO: your code here

    for i in range(0, len(bits), 8): #loops over each 8-bit 
        byte_8bits = bits[i:i+8] #create a list starting at i and ending after we reached 8th bit // we divide out bits into 8s because each byte is 8 bits
        debit_byte = int(''.join(map(str, byte_8bits)), 2) #we use str() to convert debit_byte to a string so we can use .join with '' and then after that we use int([], 2) to convert back to int in base 2/binary
        byts.append(debit_byte)

    return byts

def bit2hex(bits: Iterable[int]) -> str:
    """
    convert bits to hex string
    """
    return debitize(bits).hex()

def hex2bit(hex_str: str) -> list:
    """
    convert hex string to bits
    """
    return bitize(bytes.fromhex(hex_str))

#note: a permutation table is a list of random indexes used to shuffle
def permute(raw_seq: Iterable, table: Iterable[int]) -> list: #out raw sequence is out input and our table is our permutation table
    """
    permute bits with a table
    """
    # TODO: your code here
    permute_seq = []

    #check if lists are the same length
    if(len(raw_seq) != len(tabel)):
        raise ValueError("Invalid Index! Ensure both inputs have the same length.")

    for index in table:
        permute_seq.append(raw_seq[index])
        
    return permute_seq 

def xor(bits1: Iterable[int], bits2: Iterable[int]) -> 'list[int]':
    """
    xor two bits
    """
    # TODO: your code here
    xor_result = []

    #check if lists are the same length
    if len(bits1) != len(bits2):
        raise ValueError("Input lists are not the same length.")
    
    for bits1, bits2 in zip(bits1, bits2): #zip() to combines iterables -- for each element from bits1 and bits2 iteration
        xor_result.append(bits1 ^ bits2)  #use ^ as bitwise XOR and append the result between bits1 and bits2

    return xor_result 

class DES:

    # initial permutation
    IP = [
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7,
        56, 48, 40, 32, 24, 16, 8, 0,
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6
    ]

    # final permutation
    FP = [
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25,
        32, 0, 40, 8, 48, 16, 56, 24
    ]

    # parity-bit drop table for key schedule
    KEY_DROP = [
        56, 48, 40, 32, 24, 16, 8, 0,
        57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 60, 52, 44, 36,
        28, 20, 12, 4, 27, 19, 11, 3
    ]

    BIT_SHIFT = [
        1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
    ]

    # key compression permutation
    KEY_COMPRESSION = [
        13, 16, 10, 23, 0, 4, 2, 27,
        14, 5, 20, 9, 22, 18, 11, 3,
        25, 7, 15, 6, 26, 19, 12, 1,
        40, 51, 30, 36, 46, 54, 29, 39,
        50, 44, 32, 47, 43, 48, 38, 55,
        33, 52, 45, 41, 49, 35, 28, 31
    ]
    
    # D box, key expansion permutation
    D_EXPANSION = [
        31, 0, 1, 2, 3, 4,
        3, 4, 5, 6, 7, 8,
        7, 8, 9, 10, 11, 12,
        11, 12, 13, 14, 15, 16, 
        15, 16, 17, 18, 19, 20,
        19, 20, 21, 22, 23, 24,
        23, 24, 25, 26, 27, 28, 
        27, 28, 29, 30, 31, 0
    ]
    
    # S boxes
    S1 = [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ]

    S2 = [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ]

    S3 = [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ]

    S4 = [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ]

    S5 = [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ]

    S6 = [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ]

    S7 = [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ]

    S8 = [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
    
    # S-box substitution
    S = [S1, S2, S3, S4, S5, S6, S7, S8]
    
    # D box, straight permutation
    D_STRAIGHT = [
        15, 6, 19, 20, 28, 11, 27, 16,
        0, 14, 22, 25, 4, 17, 30, 9,
        1, 7, 23, 13, 31, 26, 2, 8,
        18, 12, 29, 5, 21, 10, 3, 24
    ]

    @staticmethod
    def key_generation(key: 'list[int]') -> 'list[list[int]]':
        """
        raw_key: 64 bits
        return: 16 * (48bits key)
        """

        keys: 'list[list[int]]' = []
        # TODO: your code here
        #use KEY_DROP tp change key from 64-bits into 58-bits
        new_key = permute(key, DES.KEY_DROP)

        #split 56-bit key into 2 28-bits left and right
        left_key = key[:28]
        right_key = key[28:]

        for round in range(16): #16 rounds of shifting

            shift_round = DES.BIT_SHIFT[round] #the round we are on - rounds 1, 2, 9, and 16 shift one bit, the rest 2 bits

            #shift left -- [round:] takes all elements to the right of round; [:round] takes the round and all elements to its left
            left_key = left_key[round:] + left_key[:round] 
            right_key = right_key[round:] + right_key[:round]

            #we combine them to effectivley make the "shift" happen
            combined_key = left_key + right_key
            #permute with KEY_COMPRESSION to change from 58-bit to 48-bit
            subkey = permute(combined_key, DES.KEY_COMPRESSION)
            keys.append(subkey)

        return keys

    @staticmethod
    def f(R: 'list[int]', key: 'list[int]') -> 'list[int]':
        """
        f function
        R: 32 bits
        key: 48 bits
        return: 32 bits
        """
        # TODO: your code here

        #use provided D_EXPANSION for the permutation 
        R_expanded = permute(R, DES.D_EXPANSION)
        #xor expanded with given round key
        xor_R = xor(R_expanded, key)

        sbox_output = []

        for i in range(8): #loop to do the mixing with 8 provided S-boxes

            six_bits = xor_R[i*6 : (i+1)*6] #operation that gets 6-bits -- we split the 48-bits into 8 6-bits

            #note: the first and last bit determine the row, and the middle four bits determine the column in the S-box
            #take the bits accordingly and combine them to create a binary string that will convert to a corresponding row/col number
            sbox_row = int(f"{six_bits[0]}{siz_bits[5]}", 2)
            sbox_col = int(f"{six_bits[1]}{six_bits[2]}{six_bits[3]}{six_bits[4]}, 2")

            #use DES.S to access the 8 different S-boxs accordinly to i -- implement row and col to get the value
            sbox_value = DES.S[i][row][col]

            #.extend() to add elements onto an iterable while we use map() -- use bin() to convert into binary string -- [2:] takes the 3rd to the last bit -- zfill(4) is padding to ensure we leave with a 4-bit
            sbox_output.extend(map(int, bin(sbox_value[2:].zfill(4))))

        #permute our output after 8 rounds of S-box with the provided D_STRAIGHT -- this is the final step for the function
        output_permute = permute(sbox_output, DES.D_STRAIGHT)

        return output_permute 

    @staticmethod  
    def mixer(L: 'list[int]', R: 'list[int]', sub_key: 'list[int]') -> 'tuple[list[int]]':
        """
        right_half: 32 bits
        sub_key: 48 bits
        return: 32 bits
        """
        # TODO: your code here
        # tips: finish f and xor first, then use them here
        #first step: expand right half using D_EXPANSION
        R_expanded = permutee(R, DES.D_EXPANSION)
        #second step: xor r_expanded with subkey
        xor_result = xor(R_expanded, sub_key)
        #third step: apply funciton f / S-box substitution
        sbox_sub = DES.f(xor_result, sub_key)
        #fourth step: xir left half and the output of sbox sub
        new_R = xor(L, sbox_sub)

        return (R, new_R) #return the right half and the new right half as tuples 
    
    @staticmethod
    def swapper(L: 'list[int]', R: 'list[int]') -> 'tuple[list[int]]':
        """
        A free function for you, LMAO ^O^
        """
        return R, L

    def __init__(self, raw_key: bytes) -> None:
        # for encryption use
        self.keys = DES.key_generation(bitize(raw_key))
        
        # for decryption use
        self.reverse_keys = deepcopy(self.keys)
        self.reverse_keys.reverse()

    def enc_block(self, block: 'list[int]') -> 'list[int]':
        """
        Encrypt a block of 64 bits (8 bytes).
        block: 64 bits.
        return: 64 bits.
        """
        # TODO: your code here

        





        return [] # just a placeholder

    def dec_block(self, block: 'list[int]') -> 'list[int]':
        """
        similar to enc_block
        block: 64 bits
        return: 64 bits
        """
        # TODO: your code here
        return [] # just a placeholder

    def encrypt(self, msg_str: str) -> bytes:
        """
        Encrypt the whole message.
        Handle block division here.
        *Inputs are guaranteed to have a length divisible by 8.
        """
        # TODO: your code here
        return b'' # just a placeholder
    
    def decrypt(self, msg_bytes: bytes) -> str:
        """
        Decrypt the whole message.
        Similar to encrypt.
        """
        # TODO: your code here
        return '' # just a placeholder