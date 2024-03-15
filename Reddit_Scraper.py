import argparse
import praw

class Reddit_Scraper():

    def __init__(self, ID, secret, user_agent):
        self.reddit = praw.Reddit(client_id = ID, client_secret = secret, user_agent = user_agent)

    def list_new_subreddits(self, given_limit):
        with open('List_Of_New_Subreddits.csv', 'w', encoding = 'utf-8') as file:
            for subreddit in self.reddit.subreddits.new(limit = given_limit):
                file.write(subreddit.display_name + '\n')

    def get_text_from(self, subreddit, given_query, given_number_of_submissions):
        with open(f'Text_From_{given_number_of_submissions}_Submissions.txt', 'w', encoding = 'utf-8') as file:
            for submission in self.reddit.subreddit(subreddit).search(given_query, limit = given_number_of_submissions):
                file.write('\n----- Begin Submission Title -----\n' + submission.title + '\n----- End Submission Title -----\n')
                if submission.selftext == '':
                    file.write('\n----- Begin Submission URL -----\n' + submission.url + '\n----- End Submission URL -----\n')
                else:
                    file.write('\n----- Begin Submission Self-Text -----\n' + submission.selftext + '\n----- End Submission Self-Text -----\n')
                submission.comments.replace_more(limit = None)
                for comment in submission.comments.list():
                    file.write('\n----- Begin Comment -----\n' + '--->'*comment.depth + comment.body + '\n----- End Comment -----\n')

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
    Reddit_scraper = Reddit_Scraper(dictionary_of_arguments['ID'], dictionary_of_arguments['secret'], dictionary_of_arguments['user_agent'])
    #Reddit_scraper.list_new_subreddits(given_limit = 40_000)
    Reddit_scraper.get_text_from(subreddit = 'all', given_query = '*', given_number_of_submissions = 5)