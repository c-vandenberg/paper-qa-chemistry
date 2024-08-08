def get_user_confirmation(prompt: str) -> bool:
    yes_responses = {'y', 'yes'}
    no_responses = {'n', 'no'}

    while True:
        response = input(prompt).strip().lower()
        if response in yes_responses:
            return True
        elif response in no_responses:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def get_user_positive_integer(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
