import argparse
import praw

def list_new_subreddits(ID, secret, user_agent):
    reddit = praw.Reddit(client_id = ID, client_secret = secret, user_agent = user_agent)
    with open('List_Of_New_Subreddits.txt', 'w') as file:
        for subreddit in reddit.subreddits.new(limit = 40_000):
            file.write(subreddit.display_name + '\n')

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'List New Subreddits', description = 'This program lists new subreddits.')
    parser.add_argument('ID', help = 'nonsense string under name and type of Reddit app')
    parser.add_argument('secret', help = "nonsense string right of word 'secret'")
    parser.add_argument('user_agent', help = 'e.g., API_Accessor/0.0.0 (by /u/Me)')
    args = parser.parse_args()
    ID = args.ID
    secret = args.secret
    user_agent = args.user_agent
    dictionary_of_arguments['ID'] = ID
    dictionary_of_arguments['secret'] = secret
    dictionary_of_arguments['user_agent'] = user_agent
    return dictionary_of_arguments

if __name__ == '__main__':
    dictionary_of_arguments = parse_arguments()
    list_new_subreddits(dictionary_of_arguments['ID'], dictionary_of_arguments['secret'], dictionary_of_arguments['user_agent'])