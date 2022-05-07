from jsonmodels import models, fields


class Sample(models.Base):
    file_location = fields.StringField()
    birth_place = fields.StringField()
    native_language = fields.StringField()
    other_languages = fields.StringField()
    age_sex = fields.StringField()
    age_of_english_onset = fields.StringField()
    english_learning_method = fields.StringField()
    english_residence = fields.StringField()
    length_of_english_residence = fields.StringField()

    def set_field(self, field_key, field_value):
        if field_key == 'birth_place':
            self.birth_place = ' '.join(field_value.split()[2:-1])
        elif field_key == 'native_language':
            self.native_language = ' '.join(field_value.split()[2:]).replace('\n', ', ')
        elif field_key == 'other_languages':
            self.other_languages = ' '.join(field_value.split()[2:])
        elif field_key == 'age_sex':
            self.age_sex = ' '.join(field_value.split()[2:])
        elif field_key == 'age_of_english_onset':
            self.age_of_english_onset = ' '.join(field_value.split()[4:])
        elif field_key == 'english_learning_method':
            self.english_learning_method = ' '.join(field_value.split()[3:])
        elif field_key == 'english_residence':
            self.english_residence = ' '.join(field_value.split()[2:])
        elif field_key == 'length_of_english_residence':
            self.length_of_english_residence = ' '.join(field_value.split()[4:])
        else:
            raise ValueError('Unexpected key: {}'.format(field_key))
