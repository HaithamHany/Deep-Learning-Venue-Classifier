def get_user_model_choice():
    while True:
        print("\nSelect the model you want to use:")
        print("1. Convolutional Neural Network (CNN)")
        print("2. Decision Tree")
        choice = input("Enter 1 or 2: ")
        if choice in ['1', '2']:
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    choice = get_user_model_choice()

    if choice == '1':
        print("You selected CNN.")
        from CNN import main_cnn
        main_cnn.run_cnn()  # Call the function
    elif choice == '2':
        print("You selected Decision Tree.")
        from DecisionTree import main_decision_tree
        main_decision_tree.run_decision_tree()  # Call the function
