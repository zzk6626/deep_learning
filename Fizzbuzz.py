# 一共有4个类别 ， 编码
# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0

# 解码
def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "bizzbuzz"][prediction]

def helper(i):
    print(fizz_buzz_decode(i, fizz_bizz_encode(i)))


