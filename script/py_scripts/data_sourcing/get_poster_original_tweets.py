import config.config as config
import helper.twarc_helper as tw_hp


if __name__ == "__main__":
    config = config.config()
    path = config['PATHS']
    conversation_ids_5 = path['conversation_ids_5']

    poster_path = config['POSTER_PATH']
    poster_original_tweets_file = poster_path['poster_original_tweets_file']
    tw_hp.get_tweets(conversation_ids_5,
                     poster_original_tweets_file)


