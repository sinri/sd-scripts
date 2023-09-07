import os

import requests
from nehushtan.logger.NehushtanFileLogger import NehushtanFileLogger

from sinri.drawer.SSDrawer import SSDrawer
from workspace.gen import env


class GathInnTask:
    def __init__(self, row: dict):
        self.__row = row

    def get_application_id(self):
        return self.__row['application_id']

    def execute(self, logger: NehushtanFileLogger):
        if self.__row['status'] != 'APPLIED':
            raise Exception('status is not APPLIED')

        output_file = self.__draw_an_image()

        if not os.path.isfile(output_file):
            raise Exception('cannot find output file')

        logger.info(f'drawn and saved to {output_file}')

        with open(output_file, 'rb') as file_to_upload:
            files = {'file': file_to_upload}
            values = env.inn_gibeah_verification

            r = requests.post(env.inn_gibeah_upload_url, files=files, data=values)
            print(f'uploaded: {r.status_code} | {r.text}')

        # remove
        os.remove(output_file)

    def __draw_an_image(self):
        if not os.path.isdir(env.inn_output_folder):
            os.mkdir(env.inn_output_folder)

        self.__drawer = SSDrawer(env=env.SinriImgGen).set_output_dir(env.inn_output_folder)

        self.__drawer.set_model(self.__row.get('model'))
        self.__drawer.set_size(self.__row.get('width'), self.__row.get('height'))
        self.__drawer.set_steps(self.__row.get('steps'))
        self.__drawer.set_sampler(self.__row.get('scheduler'))

        clip_skip = self.__row.get('clip_skip')
        if clip_skip > 0:
            self.__drawer.set_clip_skip(clip_skip)

        self.__drawer.set_seed(self.__row.get('seed'))

        self.__drawer.set_prompt(
            positive_content=self.__row.get('prompt'),
            positive_scale=self.__row.get('cfg'),
            negative_content=self.__row.get('negative_prompt'),
        )

        vae = self.__row.get('vae')
        if vae:
            self.__drawer.set_vae(vae)

        textual_inversion_rows = self.__row.get('textual_inversion_rows')
        if textual_inversion_rows is not None and len(textual_inversion_rows) > 0:
            for textual_inversion_row in textual_inversion_rows:
                self.__drawer.add_textual_inversion(textual_inversion_row['textual_inversion'])

        lora_rows = self.__row.get('lora_rows')
        for lora_row in lora_rows:
            self.__drawer.add_network(key=lora_row['lora'],network_mul=lora_row['lora_multiplier'])

        return self.__drawer.draw()
