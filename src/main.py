from preprocessing.preprocessing import preprocessing


if __name__ == "__main__":
    nr_of_cities = 30

    cities = preprocessing(30)
    
    for city in cities:
        print(city)
