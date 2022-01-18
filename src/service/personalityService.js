import {config} from "dotenv";
import utils from "../utils.js";

config()


class PersonalityService {
    static channelBotPersonality = {}

    static getChannelPersonality(channel) {
        if (!this.channelBotPersonality[channel]) {
            this.changeChannelPersonality(channel)
        }
        return this.channelBotPersonality[channel]
    }

    static changeChannelPersonality(channel, code = process.env.TRANSLATION_FILE || "en-EN", botName = process.env.BOTNAME) {
        try {
            this.channelBotPersonality[channel] = utils.loadJSONFile(`./translations/aiPersonality/${botName}/${code}.json`)
            return true
        } catch (e) {
            try {
                this.channelBotPersonality[channel] = utils.loadJSONFile(`./translations/aiPersonality/${botName}/${process.env.TRANSLATION_FILE}.json`)
            } catch (e2) {
                this.channelBotPersonality[channel] = utils.loadJSONFile(`./translations/aiPersonality/CustomAI/en-EN.json`)
            }
        }
    }
}

export default PersonalityService