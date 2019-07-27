import inflect

p = inflect.engine()


def count_occurances(arr, n, x):
    res = 0
    for i in range(n):
        if x == arr[i]:
            res += 1
    return res


def word_dictionary(sorted_words):

    word_dictionary = {}

    for i, _ in enumerate(sorted_words):

        find_word = sorted_words[i]
        length_of_list = len(sorted_words)

        result = count_occurances(sorted_words, length_of_list, find_word)
        word_dictionary[find_word] = result

    return word_dictionary


def search(arr, x):

    for i in range(len(arr)):

        if arr[i] == x:
            return i

    return -1


def pre_final_dictionary(dictionary):

    consonant = ['a', 'e', 'i', 'o', 'u']

    pre_final_dictionary = {}

    for key, value in dictionary.items():

        if value == 1:

            res = search(consonant, key[0])

            if not res == -1:

                pre_final_dictionary[key] = 'an'

            else:

                pre_final_dictionary[key] = 'a'

        else:

            pre_final_dictionary[key] = value

    return pre_final_dictionary


def final_dictionary(dictionary):

    new_dictionary = {}

    for key, value in dictionary.items():

        if isinstance(value, int):

            new_dictionary[key] = p.number_to_words(value)

        else:

            new_dictionary[key] = value

    return new_dictionary


def is_are(dictionary):

    if (len(dictionary) == 1) and (dictionary.values() == 'a' or 'an'):

        return 'is'

    elif (len(dictionary) == 1) and (not dictionary.values() == 'a' or 'an'):

        return 'are'

    else:

        return 'are'


def comma_and(key, dictionary):

    k = list(dictionary.items())

    if key == k[len(k) - 2][0]:

        return 'and'

    else:

        return ','


def comma_and_placement(dictionary):

    and_tag = []

    for key, value in dictionary.items():

        if len(dictionary) == 1:

            and_tag.append(value)
            and_tag.append(key)

        else:

            and_tag.append(value)
            and_tag.append(key)
            and_tag.append(comma_and(key, dictionary))

    if not len(dictionary) == 1:

        and_tag = and_tag[:-1]

    return and_tag
