import {config} from "dotenv";
import axios from "axios";
import utils from '../utils.js'
import messageService from "./messageService.js";
import lmiService from "./lmiService.js";
import phraseBiasService from "./phraseBiasService.js";
import bannedTokensService from "./bannedTokensService.js";
import presetService from "./presetService.js";

config()


const conf = utils.loadJSONFile("./conf.json")
let lastGenerationTimestamp = Date.now()

const DEFAULT_PARAMETERS = utils.loadJSONFile("./data/aiParameters/personality_default.json")
const DEFAULT_PARAMETERS_EVALBOT = utils.loadJSONFile("./data/aiParameters/evalbot_default.json")

const generateUnthrottled = async (input, params) => {
    let res
    console.log("generateUnthrottled started")
    try {
        res = await axios.post(
            `http://localhost:${process.env.PORT || "7319"}/generate`,
            {
                input,
                model: process.env.AI_MODEL || "euterpe-v2",
                parameters: params
            },
            {
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        )
    } catch (e) {
        console.error(e)
        res = null
    }

    return res?.data?.output
}


const generateUnthrottledCustom = async (input, params, model) => {
    let res
    try {
        res = await axios.post(
            `http://localhost:${process.env.PORT || "7319"}/generate`,
            {
                input,
                model,
                parameters: params
            },
            {
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        )
    } catch (e) {
        console.error(e)
        res = null
    }

    return res?.data?.output
}


let isProcessing = false
// throttles generation at one request per second
const generate = async function (input, params, lowPriority = false) {
    const timeStep = parseInt(conf.minTimeBetweenApiRequestsInSeconds) * 1000
    console.log("generate started")
    if (lowPriority && isProcessing) {
        return null
    } else {
        isProcessing = true
        const timeDiff = Date.now() - lastGenerationTimestamp
        lastGenerationTimestamp = Date.now()
        await utils.sleep(timeDiff < timeStep ? timeStep - timeDiff : 0)
        const res = await generateUnthrottled(input, params)
        isProcessing = false
        return res
    }
}

const generateCustom = async function (input, parameters, model) {
    const timeStep = parseInt(conf.minTimeBetweenApiRequestsInSeconds) * 1000

    isProcessing = true
    const timeDiff = Date.now() - lastGenerationTimestamp
    lastGenerationTimestamp = Date.now()
    await utils.sleep(timeDiff < timeStep ? timeStep - timeDiff : 0)
    const res = await generateUnthrottledCustom(input, parameters, model)
    isProcessing = false
    return res
}


class AiService {
    static async sendUntilSuccess(prompt, preventLMI, channel) {
        let answer
        let parsedAnswer
        let nbTry = 0
        let params = JSON.parse(JSON.stringify(DEFAULT_PARAMETERS))


        if (channel) {
            const aiParameters = presetService.getAiParameters(channel)
            if (aiParameters) {
                params = aiParameters
            }

            const bannedTokens = bannedTokensService.getBannedTokens(channel)
            if (bannedTokens && bannedTokens.length > 0)
                params.bad_words_ids = bannedTokens

            const phraseBiases = phraseBiasService.getPhraseBiases(channel)
            if (phraseBiases && phraseBiases.length > 0)
                params.logit_bias_exp = phraseBiases
        }

        params.repetition_penalty_range = prompt.repetition_penalty_range

        while (!parsedAnswer && ++nbTry <= 3) {
            answer = await this.sendPromptDefault(prompt.prompt, params)
            parsedAnswer = messageService.parse(answer)
        }

        if (!preventLMI) {
            lmiService.updateLmi(prompt.prompt, answer, parsedAnswer)
        }

        return parsedAnswer
    }

    static async sendPromptDefault(prompt, params = DEFAULT_PARAMETERS, lowPriority = false) {
        console.log("started sendPromptDefault")
        return await generate(prompt, params, lowPriority)
    }

    static async executePrompt(prompt, params, model) {
        return await generateCustom(prompt, params, model)
    }

    /**
     * Generates an answer given a prompt
     * Retries until fulfillment
     * @param prompt
     * @param tokensToGenerate
     * @param preventLMI
     * @param eos_token_id
     */
    static async simpleEvalbot(prompt, tokensToGenerate = 1, preventLMI = false, eos_token_id = 198) {
        const params = JSON.parse(JSON.stringify(presetService.getAiParameters("evalbot")))

        console.log("started SimpleEvalBot")


        params.max_length = tokensToGenerate
        params.eos_token_id = eos_token_id

        const result = await this.sendPromptDefault(prompt, params)
        const parsedResult = result
        if (!preventLMI) {
            lmiService.updateLmi(prompt, result, parsedResult)
        }
        return parsedResult
    }

    static async getAccessToken(key) {
        return await utils.getAccessToken(key)
    }
}

export default AiService