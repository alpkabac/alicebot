import {config} from "dotenv";

config()
import utils from "../utils.js";


class TranslationsService {
    static _translations

    static get translations() {
        if (!this._translations) {
            this.changeLanguage()
        }
        return this._translations
    }

    static changeLanguage(code = "en-EN") {
        try {
            this._translations = utils.loadJSONFile(`./translations/${code}.json`)
            return true
        } catch (e) {
            this._translations = utils.loadJSONFile(`./translations/en-EN.json`)
        }
        return false
    }
}

export default TranslationsService