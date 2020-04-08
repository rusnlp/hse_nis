from collections import defaultdict
from json import dump, load
import os
from telegram.ext import Updater
from telegram.ext import Filters
from telegram.ext import CommandHandler, MessageHandler, ConversationHandler
from telegram import ReplyKeyboardMarkup, KeyboardButton

from settings import token, help_text, forms_path, marked_forms_path, users_marked_path


class Bot():
    def __init__(self, token):
        self.forms_path = forms_path
        self.marked_forms_path = marked_forms_path
        self.users_marked_path = users_marked_path

        forms, marked_forms, users_marked = self.load_data()
        self.forms = forms
        self.marked_forms = marked_forms
        self.users_marked = users_marked

        self.help_text = help_text

        MainHandler = ConversationHandler(
            entry_points=[MessageHandler(Filters.text, self.handle_message)],
            states={
                'start': [MessageHandler(Filters.text, self.handle_message)], # только чтобы из помощи при старте могли вернуться с "начать"
                'new': [MessageHandler(Filters.text, self.handle_message)]
            },
            fallbacks=[CommandHandler('stop', self.stop)]
        )

        self.updater = Updater(token=token)
        self.updater.dispatcher.add_handler(CommandHandler('start', self.start))
        self.updater.dispatcher.add_handler(MainHandler)

    def load_data(self):
        try:
            os.mkdir('data')
        except OSError:
            pass

        forms = load(open(self.forms_path, encoding='utf-8'))

        if os.path.isfile(self.marked_forms_path):
            marked_forms = load(open(self.marked_forms_path, encoding='utf-8'))
        else:
            marked_forms = {i: 0 for i in forms}

        if os.path.isfile(self.users_marked_path):
            old_users_marked = load(open(self.users_marked_path, encoding='utf-8'))
            users_marked = defaultdict(list)
            for user in old_users_marked:  # сохраняем тип defaultdict
                users_marked[user] = old_users_marked[user]
        else:
            users_marked = defaultdict(list)  # chat_id: [id размеченных форм]

        return forms, marked_forms, users_marked


    def backup_data(self):
        dump(self.marked_forms, open(self.marked_forms_path, 'w', encoding='utf-8'))
        dump(self.users_marked, open(self.users_marked_path, 'w', encoding='utf-8'))

    start_markup = [[KeyboardButton('Начать'), KeyboardButton('Помощь')]]
    start_menu = ReplyKeyboardMarkup(start_markup, resize_keyboard=True, one_time_keyboard=False)

    cont_markup = [[KeyboardButton('Дальше'), KeyboardButton('Помощь')]]
    cont_menu = ReplyKeyboardMarkup(cont_markup, resize_keyboard=True, one_time_keyboard=False)

    def start(self, bot, update):
        bot.sendMessage(chat_id=update.message.chat_id, text='Здравствуйте! Приступим?)', reply_markup=self.start_menu)

    def stop(self):
        return ConversationHandler.END

    def suggest_form(self, chat_id):
        user_marked = self.users_marked[str(chat_id)]  # получаем список уже размеченных для пользователя
        print(chat_id, 'уже размечал', user_marked)
        # определяем, какая форма нуждается в разметке
        forms_in_need = sorted(self.marked_forms, key=self.marked_forms.get)
        print("В разметке нуждаются", forms_in_need)
        i = 0
        try:
            while forms_in_need[i] in user_marked:
                i += 1
            form_id = forms_in_need[i]
            form = {'form_id': form_id, 'form_link': self.forms[form_id]}
            print('Выбрали форму', form)
        except IndexError:
            form = None

        return form

    def handle_message(self, bot, update):
        print("\nReceived", update.message)
        message_text = update.message.text.lower()
        chat_id = update.message.chat_id
        react_dict = {'начать': 'Сейчас подкинем Вам статей на разметку...',
                      'дальше': 'Ещё? Отлично!'}

        if message_text == 'начать' or message_text == 'дальше':
            bot.sendMessage(chat_id=chat_id, text=react_dict[message_text])
            form = self.suggest_form(chat_id)
            if form:
                bot.sendMessage(chat_id=chat_id, text=form['form_link'])
                self.marked_forms[form['form_id']] += 1
                self.users_marked[str(chat_id)].append(form['form_id'])  # занесли в список размеченных для пользователя
                bot.sendMessage(chat_id=chat_id, text='Спасибо за разметку!', reply_markup=self.cont_menu)
                self.backup_data()
                print(chat_id, 'разметил', form['form_id'])
                return 'new'

            else:
                bot.sendMessage(chat_id=chat_id, text='Вы герой и разметили все наши формы!')
                print(chat_id, 'разметил все формы')
                return ConversationHandler.END

        if update.message.text.lower() == 'помощь':
            bot.sendMessage(chat_id=chat_id, text=self.help_text, parse_mode='Markdown', reply_markup=self.start_menu)
            return 'start'


if __name__ == "__main__":
    print('Работаем!\n')
    bot = Bot(token)
    bot.updater.start_polling()
