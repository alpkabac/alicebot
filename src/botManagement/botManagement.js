import {config} from "dotenv";

import express from "express"
import fs from "fs"
import path from "path"
import child_process from "child_process"
import {v4} from 'uuid'
import RunningBot from "./RunningBot.js"
import axios from "axios";
import aiService from "../service/aiService.js";
import generate from "../rest/generatorApi.js";
import utils from "../utils.js";
import logService from "../service/logService.js";
import njsTrace from "njstrace";

njsTrace.inject()

config()

const uuid = v4
const app = express()
app.use(express.json())
app.use(express.urlencoded({extended: true}))

const processes = {}
const accessTokens = []

try {
    fs.mkdirSync('./user')
    fs.mkdirSync('./bot')
} catch {

}

try {
    const dir = fs.readdirSync('./user')
    if (dir.length === 0) {
        const admin = {
            "username": "admin",
            "id": "admin",
            "key": "admin",
            "isAdmin": true,
            "isPatreon": false,
            "bots": [],
            "maxBots": 1000
        }
        fs.writeFileSync('./user/admin.json', JSON.stringify(admin, null, 4))
    }
} catch (e) {
    logService.error("Couldn't create admin user file", e)
}

let runningBots
try {
    runningBots = loadJSONFile('./runningBots.json') || []
} catch (e) {
    runningBots = []
}

const runningBotsCopy = JSON.parse(JSON.stringify(runningBots))
for (let runningBot of runningBotsCopy) {
    const error = startBot(runningBot.id, runningBot.preset, true)
    if (error) console.error(error)
}

handleLoadBalancing()

class User {
    constructor(username, id = uuid()) {
        this.username = username
        this.id = id
        this.key = uuid()
        this.isAdmin = false
        this.isPatreon = true
        this.bots = []
        this.maxBots = 1
    }
}

function handleError(res, message, detail = null) {
    console.error(message)
    res.json({
        status: 'ERROR',
        error: message,
        detail
    })
}

function loadJSONFile(filename) {
    let file
    try {
        file = fs.readFileSync(filename)
        return JSON.parse(file)
    } catch (err) {
        // logService.error("Couldn't load JSON file", err)
    }
}

function loadUser(id) {
    const filePath = `./user/${id}.json`
    if (fs.existsSync(filePath)) {
        const user = loadJSONFile(filePath)
        if (!user) throw new Error("Could find or parse JSON for user " + id)
        return user
    } else {
        throw new Error("No user found for id " + id)
    }
}

function loadBotData(id) {
    const filePath = `./bot/${id}/default.json`
    if (fs.existsSync(filePath)) {
        const bot = loadJSONFile(filePath)
        if (!bot) throw new Error("Couldn't find or parse JSON for bot " + id)
        return bot
    } else {
        throw new Error("No bot data found for id " + id)
    }
}

function loadBotPersonality(id) {
    const filePath = `./bot/${id}/default.personality`
    if (fs.existsSync(filePath)) {
        const bot = loadJSONFile(filePath)
        if (!bot) throw new Error("Couldn't find or parse JSON for bot " + id)
        return bot
    } else {
        throw new Error("No bot personality found for id " + id)
    }
}

function authenticate(reqBody, adminOnly = false) {
    let user
    try {
        user = loadUser(reqBody?.login)
    } catch (e) {
        throw new Error("Authentication failed")
    }

    if (adminOnly && !user.isAdmin) throw new Error("You don't have the proper permissions for this action")

    if (user.id !== reqBody?.login || user.key !== reqBody?.key) {
        throw new Error("Authentication failed")
    }

    return user
}

function userHasPermissionOverBot(user, botId) {
    if (user.isAdmin) return true
    if (!user.bots.includes(botId)) {
        throw new Error("You don't have permission over this bot")
    }
}

function getDefaultPersonality(data) {
    return {
        "noContextSentence": `Hello, my name is ${data.botName} and I'm an AI designed to answer all your questions.`,
        "context": data.botContext || '',
        "contextDm": data.botContextDm || data.botContext || '',
        "description": `[ Character: ${data.botName}; gender: ${data.botGender}${data.botPersonality ? '; ' + data.botPersonality : ''} ]`,
        "displayDescription": `${data.botDescription || data.botName}`,
        "introduction": [
            {
                "from": `${data.botName}`,
                "msg": `${data.presentationMessage || ''}`
            }
        ],
        "introductionDm": [
            {
                "from": `${data.botName}`,
                "msg": `${data.presentationMessageDm || data.presentationMessage || ''}`
            }
        ],
        "voice": {
            "languageCode": "en-US",
            "name": `en-US-Wavenet-${data.botGender ? data.botGender.toLowerCase() === "female" ? 'F' : 'B' : 'B'}`,
            "ssmlGender": `${data.botGender ? data.botGender.toLowerCase() === "female" ? 'FEMALE' : 'MALE' : 'MALE'}`
        },
        "bannedTokenFiles": "endoftext",
        "phraseBiasFiles": "body_language",
        "bad_words_ids": null,
        "logit_bias_exp": null
    }
}

function getDefaultEnv(data, id) {
    return `TOKEN=${data.discordToken}`
        + `\nBOTNAME="${data.botName}"`
        + `\nBOT_DISCORD_USERNAME="${data.botDiscordName || data.botName}"`
        + `\nBOT_DISCORD_AVATAR="${data.discordAvatarUrl || ''}"`
        + `\nTTS_VOICE_SEED="${data.ttsVoiceSeed || data.botName}"`
        + `\nTRANSLATION_FILE="default"`
        + `\nAI_MODEL="${data.aiModel || "euterpe-v0"}"`
        + `\nALLOWED_CHANNEL_NAMES="${data.channelName}"`
        + `\nBANNED_TOKENS_FILE="default"`
        + `\nPHRASE_BIASES_FILE="default"`
        + `\nLMI_URL="/"`
        + `\nLMI_PORT=`
        + `\nENABLE_AUTO_ANSWER=true`
        + `\nENABLE_SMART_ANSWER=false`
        + `\nENABLE_AUTO_MESSAGE=false`
        + `\nMIN_BOT_MESSAGE_INTERVAL=12`

        + `\nENABLE_CUSTOM_AI=true`
        + `\nENABLE_TTS=${data.enableTTS}`
        + `\nENABLE_DM=true`
        + `\nENABLE_INTRO=true`
        + `\nSEND_INTRO_TO_CHANNELS="${data.channelName}"`
        + `\nENABLE_GREET_NEW_USERS=false`

        + `\nENABLE_RPG_IMAGES=false`
        + `\nENABLE_RPG_MODE=false`
        + `\nRPG_ATTACK_COOLDOWN=20`
        + `\nRPG_SPAWN_COOLDOWN_MIN=120`
        + `\nRPG_SPAWN_COOLDOWN_MAX=120`
        + `\nRPG_RESPAWN_COOLDOWN_MIN=100000`
        + `\nRPG_RESPAWN_COOLDOWN_MAX=100000`

        + `\nALLOW_MUTE="Bot Moderator, Bot Administrator"`
        + `\nALLOW_REMEMBER="Bot Moderator, Bot Administrator"`
        + `\nALLOW_WIPE_REMEMBER="Bot Moderator, Bot Administrator"`
        + `\nALLOW_FORGET="Bot Moderator, Bot Administrator"`

        + `\nALLOW_CHANGE_LANGUAGE="Bot Moderator, Bot Administrator"`
        + `\nALLOW_SET_PERSONALITY="Bot Moderator, Bot Administrator"`
        + `\nALLOW_SET_JSON_PERSONALITY="Bot Moderator, Bot Administrator"`
        + `\nALLOW_DISPLAY_PERSONALITY="Bot Moderator, Bot Administrator"`
        + `\nALLOW_SET_VOICE="Bot Moderator, Bot Administrator"`

        + `\nALLOW_NO_CONTEXT_MESSAGE=true`
        + `\nALLOW_CONTINUE_MESSAGE=true`
        + `\nALLOW_RETRY_MESSAGE=true`
        + `\nALLOW_EDIT_MESSAGE="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_ANSWER_MESSAGE=true`
        + `\nALLOW_COMMENT_MESSAGE=true`
        + `\nALLOW_REACTIONS=true`
        + `\nALLOW_REPLY_TO_NAME=true`

        + `\nALLOW_DELETE_MESSAGE="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_PRUNE_MESSAGES="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_SET_ACTIVITY="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_EVENT_INJECTION_MESSAGE="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_PROPERTY_INJECTION_MESSAGE="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_RPG_SPAWN="Bot Moderator, Bot Administrator, Game Master"`
        + `\nALLOW_RPG_ATTACK="Bot Moderator, Bot Administrator, Game Master, RPG"`
        + `\nALLOW_PROMPT_MESSAGE="Bot Moderator, Bot Administrator"`
        + `\nALLOW_LORE_GENERATION_TOOL="Bot Moderator, Bot Administrator"`
        + `\nALLOW_RULE34="Bot Moderator, Bot Administrator"`
        + `\nALLOW_EPORNER="Bot Moderator, Bot Administrator"`
        + `\nALLOW_WIKI="Bot Moderator, Bot Administrator"`
        + `\nALLOW_DANBOORU="Bot Moderator, Bot Administrator"`
        + `\nALLOW_IMAGE_GENERATION="Bot Administrator, Bot Moderator"`

        + `\nBOT_ID="${id}"`
}

function startBot(id, preset = "default", force = false) {
    const filePath = `./bot/${id}/${preset}.env`

    if (!force && runningBots.some(rb => rb.id === id)) {
        return `Bot ${id} is already running`
    }

    if (!fs.existsSync(filePath)) return `Preset ${preset} for ${filePath} couldn't be found`

    const workerProcess = child_process.spawn("node", ["-r", "dotenv/config", "index_discord.js", `dotenv_config_path=${filePath}`])

    workerProcess.stdout.on('data', function (data) {
        process.stdout.write('stdout: ' + data)
    })

    workerProcess.stderr.on('data', function (data) {
        process.stdout.write('stderr: ' + data)
    })

    workerProcess.on('close', function (code) {
        process.stdout.write(`Subprocess ${id} has exited, exit code` + code)
        runningBots = runningBots.filter(rb => rb.id !== id)
        fs.writeFileSync('./runningBots.json', JSON.stringify(runningBots))
    })

    processes[id] = workerProcess
    if (runningBots.some(rb => rb.id === id)) {
        runningBots.forEach(rb => {
            if (rb.id === id) {
                if (rb.preset !== preset) {
                    rb.preset = preset
                    console.log("Overwrite preset!")
                }
            }
        })
    } else {
        runningBots.push(new RunningBot(id, preset))
        console.log(`Started bot ${id}!`)
    }
    fs.writeFileSync('./runningBots.json', JSON.stringify(runningBots))

    return false
}


app.post('/api/v1/test', function (req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.json('{"status": "SUCCESS"}')
})

app.post('/api/v1/admin/firstConnexion', function (req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body, true)
        const data = req.body

        if (!data.firstConnexionLogin) return handleError(res, "You need to provide the \"firstConnexionLogin\" as argument (new admin login)")
        if (!data.firstConnexionKey) return handleError(res, "You need to provide the \"firstConnexionKey\" as argument (new admin password)")
        if (!data.firstConnexionNaiKey) return handleError(res, "You need to provide the \"firstConnexionNaiKey\" as argument (NovelAI API key)")
        if (!data.naiSubTier) return handleError(res, "You need to provide the \"naiSubTier\" as argument (NovelAI API subscription tier)")

        if (!(authenticatedUser.id === 'admin' || authenticatedUser.key === "admin")) return handleError(res, `First connexion setup already done`)

        fs.rmSync('./user/admin.json')
        fs.writeFileSync(`./user/${data.firstConnexionLogin}.json`, JSON.stringify({
            "username": data.firstConnexionLogin,
            "id": data.firstConnexionLogin,
            "key": data.firstConnexionKey,
            "isAdmin": true,
            "isPatreon": false,
            "bots": [],
            "maxBots": 1000
        }, null, 4))

        if (!fs.existsSync('./.env')) {
            fs.writeFileSync('./.env', `NOVEL_AI_API_KEY="${data.firstConnexionNaiKey}"\nTOKEN_LIMIT=${["opus", "scroll"].includes(data.naiSubTier.toLowerCase()) ? '2048' : '1024'}`)
        }

        loadKeys(data.firstConnexionNaiKey).then()

        res.json({"status": "SUCCESS"})
    } catch (e) {
        logService.error("An error happened during first connexion", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/user/create', async function (req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body, true)
        const data = req.body

        if (!data.username) return handleError(res, "You need to provide the \"username\" as argument")
        if (!data.discordToken) return handleError(res, "You need to provide the \"discordToken\" as argument")

        const newUser = new User(data.username, data.username)
        fs.writeFileSync(`./user/${newUser.id}.json`, JSON.stringify(newUser))

        const id = `${data.username}-` + (data.id || uuid())

        fs.mkdirSync(`./bot/${id}`)

        data.botName = data.username + 'Bot'
        data.botDiscordName = data.botName
        data.channelName = '#' + data.username.toLowerCase() + '-bot'
        data.botDescription = `${data.username}'s personal bot.`
        data.botPersonality = ``
        data.presentationMessage = `Hello, my name is ${data.botName} and I'm an AI.`
        data.presentationMessageDm = `Hello, my name is ${data.botName} and I'm an AI.`
        data.discordAvatarUrl = `https://upload.wikimedia.org/wikipedia/fr/thumb/4/4f/Discord_Logo_sans_texte.svg/1818px-Discord_Logo_sans_texte.svg.png`
        data.aiModel = `euterpe-v2`

        fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.env`, getDefaultEnv(data, id))
        fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.personality`, JSON.stringify(getDefaultPersonality(data)))

        newUser.bots.push(id)
        fs.writeFileSync(`./user/${newUser.id}.json`, JSON.stringify(newUser))

        fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.json`, JSON.stringify({
            id,
            botName: data.botName,
            botDiscordName: data.botDiscordName || data.botName,
            discordToken: data.discordToken,
            botGender: data.botGender || 'unspecified',
            botDescription: data.botDescription,
            botPersonality: data.botPersonality,
            presentationMessage: data.presentationMessage,
            presentationMessageDm: data.presentationMessageDm,
            discordAvatarUrl: data.discordAvatarUrl,
            channelName: data.channelName,
            aiModel: data.aiModel,
            ttsVoiceSeed: data.ttsVoiceSeed || data.botName,
            enableTTS: data.enableTTS
        }))

        const params = {
            "login": newUser.id,
            "key": newUser.key,
            "discordToken": data.discordToken,
            "id": id,
            "channelName": data.channelName.substr(1),
        }

        const url = `http://localhost:${process.env.PORT || "7319"}/?login=${params.login}&key=${params.key}&channelName=${params.channelName}&botId=${params.id}&discordToken=${params.discordToken}`

        res.json({
            status: 'SUCCESS',
            data: {
                user: newUser,
                preset: data.preset || "default",
                botId: id,
                discordToken: data.discordToken,
                channelName: data.channelName,
                url
            }
        })
    } catch (e) {
        logService.error("An error happened during user creation", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/user/get', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body

        if (!data.id) return handleError(res, "You need to provide the user \"id\" as argument")

        if (!authenticatedUser.isAdmin && authenticatedUser.id !== data.id) {
            return handleError(res, "You can't access other users as you when you are not admin")
        }

        const user = loadUser(data.id)

        res.json({status: 'SUCCESS', data: {user}})
    } catch (e) {
        logService.error("An error happened during user retrieval", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/user/all', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body, true)
        const data = req.body

        const files = fs.readdirSync('./user')
        const users = []
        for (let file of files) {
            file = file.replace('.json', '')
            users.push(loadUser(file))
        }

        res.json({status: 'SUCCESS', data: {users}})
    } catch (e) {
        logService.error("An error happened during all user retrieval", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/edit', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        userHasPermissionOverBot(authenticatedUser, data.id)

        if (!data.id) return handleError(res, "You need to provide the user \"id\" as argument")
        if (!data.botName) return handleError(res, "You need to provide a \"botName\" as argument")
        if (!data.personality && !data.presentationMessage) return handleError(res, "You need to provide a \"presentationMessage\" as argument")
        if (!data.personality && !data.botGender) return handleError(res, "You need to provide a \"botGender\" as argument. It's not mandatory to be male or female, it can be whatever you want, but needs to be specified anyway")

        const id = data.id
        let botData
        try {
            botData = loadBotData(data.id)
            data.discordToken = botData.discordToken
            data.channelName = botData.channelName
        } catch (e) {
            return handleError(res, "You need to provide a \"discordToken\" and \"channelName\" as argument since this bot isn't properly initialized")
        }

        if (data.env) {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.env`, data.env + `\nBOT_ID=${id}`)    // Add bot id in env, very important
        } else {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.env`, getDefaultEnv(data, id))
        }

        if (data.personality) {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.personality`, JSON.stringify(data.personality, null, 4))
        } else {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.personality`, JSON.stringify(getDefaultPersonality(data), null, 4))
        }

        fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.json`, JSON.stringify({
            id,
            botName: data.botName,
            botDiscordName: data.botDiscordName || data.botName,
            discordToken: data.discordToken,
            botGender: data.botGender,
            botDescription: data.botDescription,
            botContext: data.personality?.context ? data.personality?.context : data.botContext || data.botContextDm,
            botContextDm: data.personality?.contextDm ? data.personality?.contextDm : data.personality?.context ? data.personality?.context : data.botContextDm,
            botPersonality: data.botPersonality,
            presentationMessage: data.presentationMessage,
            presentationMessageDm: data.presentationMessageDm,
            discordAvatarUrl: data.discordAvatarUrl,
            channelName: data.channelName,
            aiModel: data.aiModel,
            aiModule: data.aiModule,
            ttsVoiceSeed: data.ttsVoiceSeed || data.botName,
            enableTTS: data.enableTTS
        }, null, 4))

        if (data.phraseBias) {
            fs.writeFileSync(`./bot/${id}/default.bias`, data.phraseBias)
        }

        if (data.lorebook) {
            fs.writeFileSync(`./bot/${id}/default.lorebook`, data.lorebook)
        }

        if (data.bannedTokens) {
            fs.writeFileSync(`./bot/${id}/default.badwords`, data.bannedTokens)
        }

        if (data.presetFile) {
            fs.writeFileSync(`./bot/${id}/default.preset`, data.presetFile)
        }

        if (data.generator?.attack) {
            try {
                fs.mkdirSync(`./bot/${id}/generator`)
            } catch (e) {
            }
            fs.writeFileSync(`./bot/${id}/generator/attack.json`, JSON.stringify(data.generator.attack, null, 4))
        }

        if (data.generator?.enemy) {
            try {
                fs.mkdirSync(`./bot/${id}/generator`)
            } catch (e) {
            }
            fs.writeFileSync(`./bot/${id}/generator/enemy.json`, JSON.stringify(data.generator.enemy, null, 4))
        }

        res.json({
            status: 'SUCCESS',
            data: {
                id,
                message: `The preset "${data.preset || "default"}" of your bot ${id} has been edited! You need to restart or reset it for changes to apply`
            }
        })
    } catch (e) {
        logService.error("An error happened during bot edition", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/create', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body

        if (authenticatedUser.bots.length >= authenticatedUser.maxBots) return handleError(res, "You can't create a new bot, try to edit the ones you have already!")

        if (!data.env && !data.discordToken) return handleError(res, "You need to provide the \"discordToken\" as argument")
        if (!data.env && !data.channelName) return handleError(res, "You need to provide the \"channelName\" as argument")
        if (!data.botName) return handleError(res, "You need to provide a \"botName\" as argument")
        if (!data.discordAvatarUrl) return handleError(res, "You need to provide a \"discordAvatarUrl\" as argument")
        if (!data.personality && !data.presentationMessage) return handleError(res, "You need to provide a \"presentationMessage\" as argument")
        if (!data.personality && !data.botGender) return handleError(res, "You need to provide a \"botGender\" as argument. It's not mandatory to be male or female, it can be whatever you want, but needs to be specified anyway")

        const id = `${data.login}-` + (data.id || uuid())

        fs.mkdirSync(`./bot/${id}`)

        if (data.env) {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.env`, data.env + `\nBOT_ID=${id}`)    // Add bot id in env, very important
        } else {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.env`, getDefaultEnv(data, id))
        }

        if (data.personality) {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.personality`, JSON.stringify(data.personality))
        } else {
            fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.personality`, JSON.stringify(getDefaultPersonality(data)))
        }

        fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.json`, JSON.stringify({
            id,
            botName: data.botName,
            botDiscordName: data.botDiscordName || data.botName,
            discordToken: data.discordToken,
            botGender: data.botGender,
            botDescription: data.botDescription,
            botContext: data.botContext || data.botContextDm,
            botContextDm: data.botContextDm,
            botPersonality: data.botPersonality,
            presentationMessage: data.presentationMessage,
            presentationMessageDm: data.presentationMessageDm,
            discordAvatarUrl: data.discordAvatarUrl,
            channelName: data.channelName,
            aiModel: data.aiModel,
            ttsVoiceSeed: data.ttsVoiceSeed || data.botName,
            enableTTS: data.enableTTS
        }))

        if (data.phraseBias) {
            fs.writeFileSync(`./bot/${id}/default.bias`, data.phraseBias)
        }

        if (data.bannedTokens) {
            fs.writeFileSync(`./bot/${id}/default.badwords`, data.bannedTokens)
        }

        if (data.presetFile) {
            fs.writeFileSync(`./bot/${id}/default.preset`, data.presetFile)
        }

        console.log("New bot created: " + id)

        authenticatedUser.bots.push(id)
        fs.writeFileSync(`./user/${authenticatedUser.id}.json`, JSON.stringify(authenticatedUser))

        res.json({
            status: 'SUCCESS',
            data: {id, message: `The preset "${data.preset || "default"}" of your bot ${id} has been created!`}
        })
    } catch (e) {
        logService.error("An error happened during bot creation", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/init', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        fs.writeFileSync(createBotData, data)

        if (!data.env && !data.discordToken) return handleError(res, "You need to provide the \"discordToken\" as argument")
        if (!data.env && !data.channelName) return handleError(res, "You need to provide the \"channelName\" as argument")
        if (!data.botName) return handleError(res, "You need to provide a \"botName\" as argument")
        if (!data.personality && !data.presentationMessage) return handleError(res, "You need to provide a \"presentationMessage\" as argument")
        if (!data.personality && !data.botGender) return handleError(res, "You need to provide a \"botGender\" as argument. It's not mandatory to be male or female, it can be whatever you want, but needs to be specified anyway")

        const id = data.id || uuid()

        try {
            fs.mkdirSync(`./bot/${id}`)
        } catch {
        }

        fs.writeFileSync(`./bot/${id}/${data.preset || "default"}.json`, JSON.stringify({
            id,
            botName: data.botName,
            botDiscordName: data.botDiscordName || data.botName,
            discordToken: data.discordToken,
            botGender: data.botGender,
            botDescription: data.botDescription,
            botPersonality: data.botPersonality,
            presentationMessage: data.presentationMessage,
            presentationMessageDm: data.presentationMessageDm,
            discordAvatarUrl: data.discordAvatarUrl,
            channelName: data.channelName,
            aiModel: data.aiModel,
            ttsVoiceSeed: data.ttsVoiceSeed || data.botName,
            enableTTS: data.enableTTS
        }))

        res.json({
            status: 'SUCCESS',
            data: {id, message: `The preset "${data.preset || "default"}" of your bot ${id} has been initialized!`}
        })
    } catch (e) {
        logService.error("An error happened during bot initialization", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/get', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        if (!data.id) return handleError(res, "You must provide the id of the bot")
        userHasPermissionOverBot(authenticatedUser, data.id)

        let botData
        try {
            botData = loadBotData(data.id)
        } catch (e) {
            logService.error("Bot data couldn't be loaded", e)
        }

        let botPersonality
        try {
            botPersonality = loadBotPersonality(data.id)
        } catch (e) {
            logService.error("Bot personality couldn't be loaded", e)
        }

        res.json({status: 'SUCCESS', data: {botData, botPersonality}})
    } catch (e) {
        logService.error("An error happened during bot retrieval", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/give', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body, true)
        const data = req.body
        if (!data.userId) return handleError(res, "You must provide the userId")
        if (!data.botId) return handleError(res, "You must provide the botId")
        userHasPermissionOverBot(authenticatedUser, data.botId)

        const targetUser = loadUser(data.id)

        authenticatedUser.bots = authenticatedUser.bots.filter(bid => bid !== data.botId)
        targetUser.bots.push(data.botId)
        targetUser.maxBots++

        fs.writeFileSync(`./user/${targetUser.id}.json`, JSON.stringify(targetUser))
        fs.writeFileSync(`./user/${authenticatedUser.id}.json`, JSON.stringify(authenticatedUser))

        res.json({status: 'SUCCESS'})
    } catch (e) {
        logService.error("An error happened during bot owner change", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/running', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body

        const visibleRunningBots = authenticatedUser.isAdmin ? runningBots
            : runningBots.filter(rb => authenticatedUser.bots.some(aub => aub === rb.id))

        res.json({status: 'SUCCESS', data: {bots: visibleRunningBots}})
    } catch (e) {
        logService.error("An error happened while trying to retrieve running bots", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/getByDiscordToken', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body, true)
        const data = req.body

        if (!data.discordToken) return handleError(res, "You must provide the discordToken")

        const allBotFolders = fs.readdirSync('./bot')
        const allBots = allBotFolders.map(bf => utils.loadJSONFile(`./bot/${bf}/default.json`))

        let used = false
        let botUsingThisDiscordToken = null

        const botTokens = allBots.map(b => {
            if (data.discordToken === b.discordToken) {
                used = true
                botUsingThisDiscordToken = b
            }
            return b.discordToken
        })

        const duplicates = utils.findDuplicates(botTokens)
        const warnings = `WARNING: Some discord tokens are used multiple times!`

        res.json({
            status: 'SUCCESS',
            data: {
                used,
                botUsingThisDiscordToken: used ? botUsingThisDiscordToken : null,
                warnings: duplicates.length > 0 ? warnings : null,
                duplicates: duplicates.length > 0 ? duplicates : null
            }
        })
    } catch (e) {
        logService.error("An error happened during bot retrieval by discord token", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/start', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        if (!data.id) return handleError(res, "/api/v1/bot/start: You must provide the id of the bot")
        userHasPermissionOverBot(authenticatedUser, data.id)

        const error = startBot(data.id, data.preset || "default")
        if (error) return handleError(res, error)

        console.log(`Started bot ${data.id} preset ${data.preset || "default"}`)
        res.json({status: 'SUCCESS', data: {id: data.id, message: "Bot started!"}})
    } catch (e) {
        logService.error("An error happened during bot start", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/restart', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        if (!data.id) return handleError(res, "/api/v1/bot/stop: You must provide the id of the bot")
        userHasPermissionOverBot(authenticatedUser, data.id)

        stopBot(data.id)
        setTimeout(() => {
            const error = startBot(data.id, data.preset || "default")
            if (error) return handleError(res, error)
            res.json({status: 'SUCCESS', data: {id: data.id, message: "Bot restarted!"}})
        }, 3000)
    } catch (e) {
        logService.error("An error happened during bot restart", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/update', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        if (!data.id) return handleError(res, "/api/v1/bot/stop: You must provide the id of the bot")
        userHasPermissionOverBot(authenticatedUser, data.id)

        stopBot(data.id)
        setTimeout(() => {
            try {
                const directory = `./bot/${data.id}/save`

                fs.readdir(directory, (err, files) => {
                    if (err) console.error(err);
                    else
                        for (const file of files) {
                            fs.unlink(path.join(directory, file), err => {
                                if (err) console.error(err);
                            });
                        }
                })
            } catch (e) {
                logService.error("An error happened during bot history deletion (bot update)", e)
            }

            setTimeout(() => {
                const error = startBot(data.id, data.preset || "default")
                if (error) return handleError(res, error)
                console.log(`Bot ${data.id} updated with preset ${data.preset || "default"}!`)
                res.json({status: 'SUCCESS', data: {id: data.id, message: "Bot updated!"}})
            }, 1000)
        }, 2500)
    } catch (e) {
        logService.error("An error happened during bot update", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

app.post('/api/v1/bot/stop', async function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        const authenticatedUser = authenticate(req.body)
        const data = req.body
        if (!data.id) return handleError(res, "/api/v1/bot/stop: You must provide the id of the bot")
        userHasPermissionOverBot(authenticatedUser, data.id)

        stopBot(data.id)

        res.json({status: 'SUCCESS', data: {id: data.id, message: "Bot stopped!"}})
    } catch (e) {
        logService.error("An error happened during bot stopping", e)
        res.json({status: 'ERROR', error: e.message})
    }
})

async function loadKeys(args) {
    if (!process.env.NOVEL_AI_API_KEY && !args) return console.log("Couldn't load NovelAI API KEY")

    const keys = (args ? args : process.env.NOVEL_AI_API_KEY).split(';').map(s => s.trim())

    for (let key of keys) {
        try {
            const token = await aiService.getAccessToken(key)
            if (!accessTokens.includes(token))
                accessTokens.push(token)
        } catch (e){
            console.error(e)
        }
    }
}

async function handleLoadBalancing() {
    await loadKeys()
    let counter = 0

    app.post('/generate', async function (req, res, next) {
        res.setHeader('Access-Control-Allow-Origin', '*')

        proxy(
            req.body.input,
            req.body.model,
            req.body.parameters,
            accessTokens[counter++ % accessTokens.length]
        ).then(r => res.json(r))
    })
}

async function proxy(input, model, params, accessToken) {
    let res
    console.log("startedProxy")
    try {
        res = await axios.post(
            (process.env.API_URL || "https://api.novelai.net/ai") + "/generate",
            {
                input,
                model,
                parameters: params
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': "Bearer " + accessToken
                }
            }
        )
    } catch (e) {
        logService.error("An error happened with the custom proxy", e)
        res = null
    }
    console.log(res.data)
    return res?.data
}

async function getIndexPage(req, res){
    res.setHeader('Access-Control-Allow-Origin', '*')
    try {
        res.sendFile(path.join(path.resolve(), '/index.html'))
    } catch (e) {
        logService.error("An error happened", e)
        res.json({status: 'ERROR', error: e.message})
    }
}

app.get('/', getIndexPage)
app.get('/botManagement', getIndexPage)

app.post('/api/v1/generator', async function (req, res, next) {
    try {
        res.json(await generate(req.body))
    } catch (e) {
        res.json({status: 'ERROR', error: e})
    }
})

function stopBot(id) {
    processes[id]?.kill()
    delete processes[id]
    runningBots = runningBots.filter(rb => rb.id !== id)
    fs.writeFileSync('./runningBots.json', JSON.stringify(runningBots))
}

app.listen(parseInt(process.env.PORT || "7319"))