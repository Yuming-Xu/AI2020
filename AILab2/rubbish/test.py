def test(dic):
	print(dic)

def test_wrap(dic):
	print(dic)
	test(dic)


if __name__ == '__main__':
	tst = {1:1,2:2}
	test_wrap(tst)



