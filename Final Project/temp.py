import os


def display_image(image):
    for i in image:
        print(i)

__location__ = os.path.dirname(__file__)
f = open(os.path.join(__location__, 'data\\digitdata\\trainingimages'))
temp = f.readlines()
display_image(temp[0:20])
#temp.reverse()
# items = []
# for i in range(5):
#     data = []
#     for j in range(28):
#         data.append(list(temp.pop()))
#     if len(data[0]) < 27:
#         print('Truncating at %d examples (maximum)' % i)
#         break
#     items.append(data)
# for item in items:
#     print(item)
f.close()