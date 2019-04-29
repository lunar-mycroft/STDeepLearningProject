from recomender import Recomender

if __name__ == "__main__":
    recomender = Recomender('model-240.meta') #TODO: insert model path
    while True:
        visitorID = input("Please enter a vistor id (or quit to exit): ")
        if str(visitorID).lower() == "quit":
            break
        if not recomender.hasVisitor(visitorID):
            print("Please enter a valid user")
            continue
        k = None
        while k is None:
            k = input("Please enter the number of items to recomend: ")
            if not (isinstance(k, int) and 100>=k and k>0):
                print("Please enter an integer between 1 and 100")
                k = None
        print(recomender.makeRecomendations(visitorID,k))
