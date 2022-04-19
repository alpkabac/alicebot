import dotenv from 'dotenv'
import utils from "../utils.js"
import {Client, MessageAttachment, MessageEmbed} from 'discord.js'
import '../discord/ExtAPIMessage.js'
import {getVoiceConnection} from "@discordjs/voice"
import savingService from "../service/savingService.js"
import messageCommands from "../command/messageCommands.js"
import historyService from "../service/historyService.js"
import encoder from "gpt-3-encoder"
import aiService from "../service/aiService.js"
import promptService from "../service/promptService.js"
import updateBotInfo from "../discord/discordUtils.js"
import channelBotTranslationService from "../service/personalityService.js"
import greetingService from "../service/greetingService.js"
import commands from "../command/commands.js"
import discordCommands from "../discord/command/discordCommands.js"
import envService from "../util/envService.js"
import pawnService from "../service/rpg/pawnService.js"
import duckHuntService from "../service/rpg/duckHuntService.js"
import muteService from "../service/muteService.js"
import duckHuntCommands from "../command/duckHuntCommands.js"
import generatorService from "../service/generatorService.js";
import sharp from "sharp";
import travellingMerchantService from "../service/rpg/travellingMerchantService.js";
//import express import bodyparser
import express from 'express'
import bodyParser from 'body-parser'
import fs from "fs"
const port = 3001
//create express app
const app = express()
app.use(bodyParser.json())

dotenv.config()

const allowedCommands = []
    .concat(duckHuntCommands.attack.commandsStartsWith)
    .concat(duckHuntCommands.heal.commandsStartsWith)
    .concat(duckHuntCommands.take.commandsStartsWith)
    .concat(duckHuntCommands.drop.commandsStartsWith)
    .concat(duckHuntCommands.look.commandsStartsWith)
    .concat(duckHuntCommands.sell.commandsStartsWith)
    .concat(duckHuntCommands.equipWeapon.commandsStartsWith)
    .concat(duckHuntCommands.equipArmor.commandsStartsWith)
    .concat(duckHuntCommands.equipAccessory.commandsStartsWith)
    .concat(duckHuntCommands.equipHeal.commandsStartsWith)
    .concat(duckHuntCommands.unequipWeapon.commandsStartsWith)
    .concat(duckHuntCommands.unequipArmor.commandsStartsWith)
    .concat(duckHuntCommands.unequipAccessory.commandsStartsWith)
    .concat(duckHuntCommands.unequipHeal.commandsStartsWith)
    .concat(duckHuntCommands.showInventory.commandsStartsWith)
    .concat(duckHuntCommands.upgradeBackpack.commandsStartsWith)
    .concat(duckHuntCommands.setGender.commandsStartsWith)

const allowedCommandsMap = {
    attack: duckHuntCommands.attack.commandsStartsWith,
    heal: duckHuntCommands.heal.commandsStartsWith,
    take: duckHuntCommands.take.commandsStartsWith,
    drop: duckHuntCommands.drop.commandsStartsWith,
    look: duckHuntCommands.look.commandsStartsWith,
    sell: duckHuntCommands.sell.commandsStartsWith,
    equipWeapon: duckHuntCommands.equipWeapon.commandsStartsWith,
    equipArmor: duckHuntCommands.equipArmor.commandsStartsWith,
    equipAccessory: duckHuntCommands.equipAccessory.commandsStartsWith,
    equipHeal: duckHuntCommands.equipHeal.commandsStartsWith,
    unequipWeapon: duckHuntCommands.unequipWeapon.commandsStartsWith,
    unequipArmor: duckHuntCommands.unequipArmor.commandsStartsWith,
    unequipAccessory: duckHuntCommands.unequipAccessory.commandsStartsWith,
    unequipHeal: duckHuntCommands.unequipHeal.commandsStartsWith,
    showInventory: duckHuntCommands.showInventory.commandsStartsWith,
    upgradeBackpack: duckHuntCommands.upgradeBackpack.commandsStartsWith,
    setGender: duckHuntCommands.setGender.commandsStartsWith,
}
const bot = new Client({
    allowedMentions: {
        // set repliedUser value to `false` to turn off the mention by default
        repliedUser: false
    }
});

bot.login(process.env.TOKEN)
const channels = []
let locked = {}

let connection
let voiceChannel
let speak = null
let ttsEnabled = false


bot.on('ready', async () => {
    console.info(`Logged in as ${bot.user.tag}!`)

    if (process.env.BOT_DISCORD_USERNAME) {
        try {
            await bot.user.setUsername(process.env.BOT_DISCORD_USERNAME)
        } catch (e) {
            console.error("Couldn't rename the bot")
            try {
                await bot.user.setUsername(process.env.BOTNAME + 'Bot')
            } catch (e) {
                process.env.BOT_DISCORD_USERNAME = bot.user.tag.replace(/#.*$/, "")
            }
        }
    } else {
        try {
            await bot.user.setUsername(process.env.BOTNAME)
        } catch (e) {
            console.error("Couldn't rename the bot")
            try {
                await bot.user.setUsername(process.env.BOTNAME + 'Bot')
            } catch (e) {
                process.env.BOT_DISCORD_USERNAME = bot.user.tag.replace(/#.*$/, "")
            }
        }
    }

    if (process.env.BOT_DISCORD_AVATAR && bot.user.displayAvatarURL() !== process.env.BOT_DISCORD_AVATAR) {
        try {
            await bot.user.setAvatar(process.env.BOT_DISCORD_AVATAR)
        } catch (e) {
            console.error("Couldn't change the bot's avatar")
        }
    }

    savingService.loadAllChannels()

    if (process.env.DISCORD_ACTIVITY_NAME) {
        const name = process.env.DISCORD_ACTIVITY_NAME
        const type = process.env.DISCORD_ACTIVITY_TYPE || "PLAYING"
        await bot.user.setActivity(name, {type})
    } else {
        await bot.user.setActivity()
    }

    speak = async function (msg, channel) {
        if (!utils.getBoolFromString(process.env.ENABLE_TTS)) return console.log("TTS not enabled in env file")
        if (!ttsEnabled) return console.log("TTS NOT ENABLED")
        if (!voiceChannel) return console.log("No voice channel")
        if (channel.startsWith("##")) return console.log("Can't use TTS in DMs")

        const parsedMessage = msg
            .replace(/ \(/g, '. ')
            .replace(/\)/g, '')
            .replace(/:/g, '.')

        if (connection) {
            await utils.tts2(connection, parsedMessage)
        } else {
            console.log("Can't use TTS: No connection to vocal channel is open.")
        }
    }

    if (process.env.LOAD_CHANNEL_PERSONALITIES) {
        const personalityCodes = process.env.LOAD_CHANNEL_PERSONALITIES
            .split(",")
            .map(p => {
                const [channelName, personalityCode] = p.split(':')
                return {channelName, personalityCode}
            })

        personalityCodes.forEach(pc => {
            channelBotTranslationService.changeChannelPersonality(pc.channelName, pc.personalityCode)
        })
    }

    if (utils.getBoolFromString(process.env.ENABLE_INTRO)) {
        if (!process.env.SEND_INTRO_TO_CHANNELS) return
        const introChannels = process.env.SEND_INTRO_TO_CHANNELS
            .split(",")
            .map(v => v.trim())

        bot.channels.cache.forEach(c => {
            if (introChannels.includes(`#${c.name.toLowerCase()}`)) {
                if (envService.isRpgModeEnabled()) {
                    if (c.send) {
                        c.send('https://tenor.com/view/huzzah-adventure-jovel-haver-gif-19701828')
                    }
                } else {
                    if (historyService.getChannelHistory(`#${c.name.toLowerCase()}`).length === 0)
                        if (channelBotTranslationService.getChannelPersonality("#" + c.name.toLowerCase())?.introduction.length > 0)
                            c.send?.(utils.replaceAsterisksByBackQuotes(
                                `${channelBotTranslationService.getChannelPersonality("#" + c.name.toLowerCase()).introduction[0].msg}`
                            ))
                                .catch(() => null)
                }
            }
        })

    }

    updateBotInfo(bot)
});

//Greeting feature
bot.on('guildMemberAdd', async (member) => {
    if (!utils.getBoolFromString(process.env.ENABLE_GREET_NEW_USERS)) return
    const channel = member.guild.channels.cache.find(channel => {
        return ("#" + channel.name) === process.env.GREET_NEW_USERS_IN_CHANNEL
    })

    if (!channel) return

    const prompt = greetingService.getPrompt(member.user.username, process.env.BOTNAME)
    const message = await aiService.sendUntilSuccess({prompt}, false, "#" + channel.name)
    const parsedMessage = utils.replaceAsterisksByBackQuotes(message)
    if (parsedMessage)
        channel.send(parsedMessage).catch(() => null)
});

function replaceAliasesInMessage(message) {
    if (!process.env.BOT_DISCORD_USERNAME) return message

    return message
        .replace(process.env.BOT_DISCORD_USERNAME, process.env.BOTNAME)
        .replace(process.env.BOT_DISCORD_USERNAME.toLowerCase(), process.env.BOTNAME)
        .replace(process.env.BOT_DISCORD_USERNAME.toUpperCase(), process.env.BOTNAME)
}

const messageList = []

function appendMessage(msg) {
    messageList.push(msg)
    // const messageFormat = JSON.stringify(msg)
    // fs.writeFileSync("messageFormat", messageFormat)
}

function isMessageAnAllowedCommand(msg) {
    return allowedCommands.some(ac => msg.toLowerCase().trim().startsWith(ac.toLowerCase()))
}

function clearRpgBotTextOutput(text) {
    const isCommand = text.startsWith('!')
    let cleanContent = (isCommand ? '!' : '')
        + text.substr(isCommand ? 1 : 0)
    //.replace(/![a-zA-Z0-9*'\]].*$/, '')

    if (isCommand) {
        // removes commands if they are not the first word
        cleanContent = '!' + cleanContent.substr(1)
            .replace(/!.*$/, '')
            .replace(/\[.*$/, '')
            .replace(/;.*$/, '')
            .replace(/>.*$/, '')
            .replace(/:.*$/, '')
            .replace(/\/.*$/, '')
            .replace(/\\.*$/, '')
            .replace(/\*.*$/, '')
        //.replace(/ .*$/, '')  // removes arguments from commands

        if (isMessageAnAllowedCommand(cleanContent)) {
            let command = null
            for (let c in allowedCommandsMap) {
                if (allowedCommandsMap[c].some(ac => cleanContent.toLowerCase()?.trim()?.startsWith(ac))) {
                    command = c
                    break
                }
            }

            if (command) {
                return cleanContent // allowedCommandsMap[command][0]
            }
        }
    }

    return cleanContent
}

function colorId(id) {
    const colors = [31, 32, 35, 36, 91, 92, 93, 94, 95, 96]
    const backgrounds = [41, 42, 45, 46, 101, 102, 103, 104, 105, 106]
    return `${id.split('').map(v => `\x1b[${colors[parseInt(v)]}m\x1b[${backgrounds[parseInt(v)]}m${v}`).join('')}\x1b[0m`
}

function colorBotName(botName) {
    const colors = [31, 32, 35, 36, 91, 92, 93, 94, 95, 96]
    const color = botName.split('').map(c => c.charCodeAt(0)).reduce((partialSum, a) => partialSum + a, 0)
    return `\x1b[${colors[color % 10]}m${botName}\x1b[0m`
}

async function processMessage(msg) {
    const privateMessage = msg.channel.type === "dm"
    if (privateMessage && (!process.env.ENABLE_DM || process.env.ENABLE_DM.toLowerCase() !== "true")) return
    if (privateMessage && msg.author.username !== bot.user.username) {
        const date = new Date()
        console.log(`[${((date.getHours() < 10) ? "0" : "") + date.getHours() + ":" + ((date.getMinutes() < 10) ? "0" : "") + date.getMinutes() + ":" + ((date.getSeconds() < 10) ? "0" : "") + date.getSeconds()}]`
            + ` User ${colorId(msg.author.id)} sent a DM to ${colorBotName(process.env.BOTNAME)}`)
    }

    const channelName = privateMessage ?
        "##" + msg.channel.id
        : "#" + msg.channel.name
        
        let file 
        if(!msg.attachments.size) {
            file = null
        }
        else {
            file = msg.attachments.first().url 
        }

    if (!utils.isMessageFromAllowedChannel(channelName)) return

    const originalMsg = msg

    if (!channels[channelName])
        channels[channelName] = originalMsg.channel

    let cleanContent = replaceAliasesInMessage(utils.replaceBackQuotesByAsterisks(originalMsg.cleanContent))

    // Prevents messages from the bot itself
    // Also cache the last bot message for later retries
    if (originalMsg.author.username === bot.user.username) {
        channels[channelName].lastBotMessage = originalMsg

        if (!envService.isRpgModeEnabled()) return

        cleanContent = clearRpgBotTextOutput(cleanContent)

        const isCommandAllowed = isMessageAnAllowedCommand(cleanContent)

        if (cleanContent.startsWith('!') && !isCommandAllowed) {
            await originalMsg?.delete().catch(e => console.error(e))
            return
        }
        if (!cleanContent.startsWith('!')) {
            return
        }
    }

    if (originalMsg.content === ";ai me") return    // Prevents commands from @Patos bots

    const userRoles = originalMsg.member?.roles?.cache.map(r => {
        return {id: r.id, name: r.name}
    }) || []


    locked[channelName] = true

    let message
    if (utils.isMessageFromAllowedChannel(channelName)) {
        const allCommands = discordCommands.getOnMessageCommands()
            .concat(commands.getOnMessageCommands())
        for (let command of allCommands) {
            const commandResult = await command.call(utils.replaceNickByBotName(cleanContent).trim(), originalMsg.author.username, channelName, userRoles, originalMsg.id, bot, file)
            if (commandResult) {
                message = commandResult
                break
            }
        }
    }

    if (!message && !cleanContent.startsWith("#") && utils.getBoolFromString(process.env.ENABLE_SMART_ANSWER)
        && originalMsg.author.username !== bot.user.username
        && !cleanContent.startsWith("!")
        && !cleanContent.startsWith("#")) {
        if (await isNextMessageFromBot(channelName)) {
            await forceTalk(channelName)
        }
    }

    savingService.save(channelName)

    if (!message) {
        locked[channelName] = false
        return
    }

    if (message && message.permissionError) {
        await originalMsg.react("⛔").catch(() => null)
        return
    }


    if (message && message.error) {
        await originalMsg.react("❌").catch(() => null)
        const newMessage = await originalMsg.inlineReply(message.error).catch(() => null)
        if (newMessage && message.deleteNewMessage) {
            setTimeout(
                () => newMessage.delete().catch(() => null),
                5000
            )
        }
    }

    if (message && message.enableTTS) {
        try {
            connection.disconnect()
        } catch {
        }

        voiceChannel = msg.member?.voice?.channel


        if (!voiceChannel) {
            await originalMsg.inlineReply("# Sorry, but you need to be connected to a voice channel for me to know which one to join!")
        } else {
            ttsEnabled = true
            connection = getVoiceConnection(voiceChannel?.guild?.id)

            connection = bot.voice.connections.find((vc) => vc.channel.id === voiceChannel.id)
            if (!connection) {
                connection = await voiceChannel.join()
            }
        }
    }

    if (message && message.disableTTS) {
        ttsEnabled = false
        connection.disconnect()
        connection = null
    }

    if (message && message.success) {
        console.log("ok")
        // await originalMsg.react("✅").catch(() => null)
    }

    if (message && message.reactWith) {
        if (message.reactWith instanceof Array && message.reactWith.length > 1) {
            for (let i = 0; i < message.reactWith.length; i++) {
                setTimeout(() => {
                    originalMsg.react(message.reactWith[i]).catch(() => null)
                }, i * 250)
            }
        } else {
            await originalMsg.react(message.reactWith).catch(() => null)
        }
    }

    if (message && message.setActivityType && message.setActivityName) {
        await bot.user.setActivity(message.setActivityName, {type: message.setActivityType})
    } else if (message && message.setActivityName) {
        await bot.user.setActivity(message.setActivityName)
    }

    if (message && message.message) {
        let parsedMessage
        let messageLength
        let embedMessage = false
        try {
            messageLength = encoder.encode(message.message).length
            parsedMessage = utils.replaceAsterisksByBackQuotes(message.message)
        } catch (e) {
            messageLength = 1
            parsedMessage = message.message
            embedMessage = true
        }

        const timeToWait = message.instantReply ? 0 : messageLength * (message.fastTyping ? 10 : 50)
        if (timeToWait > 0) {
            // channels[channelName].startTyping().then()
            console.log("Typing...")
        }

        await utils.sleep(timeToWait)
        let newMessage = originalMsg

        if (message.editMessage) {
            await (await channels[channelName].messages.fetch(message.editMessage))?.edit(parsedMessage)?.catch(() => null)
        } else if (message.editLastMessage) {
            await channels[channelName].lastBotMessage?.edit(parsedMessage)?.catch(() => null)
        } else if (message.appendToLastMessage) {
            await channels[channelName].lastBotMessage?.edit(channels[channelName].lastBotMessage.cleanContent + parsedMessage)?.catch(() => null)
        } else {
            if (message.image) {
                newMessage = await originalMsg.inlineReply({
                    message: parsedMessage, files: [{
                        attachment: message.image,
                        name: "SPOILER_FILE.jpg"
                    }]
                }).catch((e) => console.error(e))
            } else {
                if (embedMessage) {
                    newMessage = await channels[channelName].send(parsedMessage).catch(() => null)
                } else {
                    await originalMsg.inlineReply(parsedMessage).catch(() => null)
                }
            }
        }

        if (!newMessage) {
            console.warn("New message couldn't be retrieved when it should have been...")
        }

        if (message && message.pushIntoHistory) {
            historyService.pushIntoHistory(message.pushIntoHistory[0], message.pushIntoHistory[1], message.pushIntoHistory[2], newMessage?.id)
        }

        if (newMessage && message.deleteNewMessage) {
            setTimeout(async () => {
                await newMessage.delete().catch(() => null)
            }, 5000)
        }

        channels[channelName].stopTyping(true)
        locked[channelName] = false
        savingService.save(channelName)

        if (message.alsoSend) {
            if (Array.isArray(message.alsoSend)) {
                for (let alsoSend of message.alsoSend) {
                    await channels[channelName].send(alsoSend).catch(e => console.error(e))
                }
            } else {
                await channels[channelName].send(message.alsoSend).catch(e => console.error(e))
            }
        }


        const shouldSpeak = (!embedMessage || message.ttsMessage) && speak && !message?.message?.startsWith?.("#") && !channelName.startsWith("##") && ttsEnabled
        if (shouldSpeak) {
            await speak(message.ttsMessage ? message.ttsMessage : parsedMessage, channelName)
        }
    } else {
        locked[channelName] = false
    }

    if (message && message.deleteMessage) {
        const m = await originalMsg?.channel?.messages?.fetch(message.deleteMessage).catch(() => null)
        if (m) {
            setTimeout(() => {
                m.delete().catch(() => null)
            }, 2000)
        } else {
            console.log("Weird...")
        }
    }

    if (message && message.deleteMessagesUpTo) {
        const allMessages = await originalMsg.channel.messages.fetch()
        const targetMessage = allMessages.find((m) => m.id === message.deleteMessagesUpTo)
        if (targetMessage) {
            setTimeout(async () => {
                const messagesToDelete = allMessages.filter(m => m.createdTimestamp >= targetMessage.createdTimestamp)

                // Synchronize history by removing newer messages
                historyService.channelHistories[channelName] = historyService.channelHistories[channelName].filter(m => {
                    const fetchedMessage = allMessages.find(_ => _.id === m.messageId)
                    return !(fetchedMessage && fetchedMessage.createdTimestamp >= targetMessage.createdTimestamp);
                })

                if (originalMsg.channel?.bulkDelete) {
                    await originalMsg.channel?.bulkDelete(messagesToDelete).catch(() => null)
                }
            }, 2000)
        }
    }

    if (message && message.deleteUserMsg) {
        setTimeout(() => {
            originalMsg.delete().catch(() => null)
        }, 2000)
    }
}

async function isNextMessageFromBot(channel) {
    const tokenCount = Math.min(process.env.TOKEN_LIMIT === "2048" ? 150 : 100, encoder.encode(process.env.BOTNAME).length)
    const promptLines = promptService.getPrompt(channel).prompt.split('\n')
    const prompt = promptLines.slice(0, -1).join('\n') + "\n"
    const result = await aiService.simpleEvalbot(prompt, tokenCount, channel.startsWith("##"))
    return !!(result && result.toLowerCase().trim() === process.env.BOTNAME.toLowerCase().trim())
}

async function forceTalk(channel) {
    const msg = await messageCommands.forceTalk.call('!talk', null, channel, [], undefined, bot)
    // If normal answer
    if (msg && msg.message?.trim()) {
        await parseForceMessage(channel, msg)
    }
}

async function parseForceMessage(channel, msg) {
    const parsedMessage = utils.replaceAsterisksByBackQuotes(msg.message)
    const timeToWait = encoder.encode(parsedMessage).length * 50
    channels[channel].startTyping().then()
    await utils.sleep(timeToWait)
    const m = await channels[channel].send(clearRpgBotTextOutput(parsedMessage)).catch((e) => console.error(e))
    if (msg.pushIntoHistory && (!msg.pushIntoHistory[0].startsWith('!') || isMessageAnAllowedCommand(msg.pushIntoHistory[0]))) {
        historyService.pushIntoHistory(clearRpgBotTextOutput(msg.pushIntoHistory[0]), msg.pushIntoHistory[1], msg.pushIntoHistory[2], m?.id)
        savingService.save(channel)
    }
    channels[channel].stopTyping(true)
    locked[channel] = false
    // if (!channel.startsWith("##") && !parsedMessage.startsWith("#") && ttsEnabled) {
    //     await speak(parsedMessage, channel)
    // }
}

async function messageLoop() {
    let msg = messageList.shift()
    while (msg) {
        await processMessage(msg)
        msg = messageList.shift()
    }

    // Auto answer
    if (utils.getBoolFromString(process.env.ENABLE_AUTO_ANSWER)) {
        for (let channel in channels) {
            // Present sending double messages
            if (locked[channel]) continue
            if (muteService.isChannelMuted(channel)) continue

            locked[channel] = true
            const msg = await messageCommands.talk.call(null, null, channel, [], undefined, bot)
            // If normal answer
            if (msg && msg.message?.trim()) {
                await parseForceMessage(channel, msg)
            } else {
                locked[channel] = false
            }
        }
    }

    // Auto message
    if (utils.getBoolFromString(process.env.ENABLE_AUTO_MESSAGE)) {
        for (let channel in channels) {

            // Prevents auto messages in DMs (temporary, hopefully)
            if (channel.startsWith("##")) continue
            if (muteService.isChannelMuted(channel)) continue
            if (locked[channel]) continue

            const history = historyService.getChannelHistory(channel)

            // Checks if the conditions for new message are met
            const historyIsEmpty = history.length === 0
            const lastMessage = historyIsEmpty ? null : history[history.length - 1]
            const timePassed = Date.now() - (parseInt(process.env.INTERVAL_AUTO_MESSAGE_CHECK || "60") * 1000)
            const enoughPassedTime = timePassed > lastMessage?.timestamp
            const isLastMessageFromBot = lastMessage?.from?.toLowerCase() === process.env.BOTNAME.toLowerCase()

            if (historyIsEmpty || !enoughPassedTime || !isLastMessageFromBot) {
                continue
            }

            locked[channel] = true
            if (await isNextMessageFromBot(channel)) {
                await forceTalk(channel)
            }
            locked[channel] = false
        }
    }

    setTimeout(messageLoop, 2500)
}

//express create post route 
app.post('/message', async, (req, res) => {
appendMessage(req.body)
const response = await messageLoop()

})

bot.on('message', appendMessage);

//express


bot.on('messageReactionAdd', async (reaction, user) => {
    if (!reaction.me) {
        const privateMessage = reaction.message?.channel?.type === "dm"
        const channelName = privateMessage ? '##' + reaction.message?.channel?.id : `#` + reaction.message?.channel?.name

        if (reaction._emoji.name === '❌') {
            historyService.delete(channelName, reaction.message?.id)
            setTimeout(async () => {
                await reaction.message?.delete().catch(() => null)
            }, 1000)
        } else if (reaction._emoji.name === '🚮') {
            historyService.prune(channelName, reaction.message?.id, reaction.message?.createdTimestamp)

            const allMessages = await reaction.message.channel.messages.fetch()
            const targetMessage = allMessages.find((m) => m.id === reaction.message?.id)
            if (targetMessage) {
                setTimeout(async () => {
                    const messagesToDelete = allMessages
                        .filter(m =>
                            (m.createdTimestamp >= targetMessage.createdTimestamp)
                            || (m.id === targetMessage.id)
                        )

                    // Synchronize history by removing newer messages
                    historyService.channelHistories[channelName] = historyService.channelHistories[channelName]
                        ?.filter?.(m => {
                            const fetchedMessage = allMessages.find(_ => _.id === m.messageId)
                            return (fetchedMessage?.createdTimestamp < targetMessage.createdTimestamp)
                                || (m.id === targetMessage.id)
                        })

                    if (reaction.message.channel?.bulkDelete) {
                        await reaction.message.channel?.bulkDelete(messagesToDelete).catch(() => null)
                    }
                }, 1000)
            }

        }
    }
});

async function mainRpgLoop() {
    if (envService.isRpgModeEnabled()) {
        while (true) {
            for (let channel in channels) {
                // Prevents auto messages in DMs (temporary, hopefully)
                if (channel.startsWith("##")) continue
                if (muteService.isChannelMuted(channel)) continue

                const swarm = duckHuntService.getSwarm()
                const swarmMode = !swarm.timestamp ? false : Date.now() - (swarm.timestamp + swarm.duration) < 0
                const pawn = pawnService.getActivePawn(channel)

                const travellingMerchantMessage = await travellingMerchantService.manageMerchant(channel)
                if (travellingMerchantMessage){
                    await channels[channel].send(travellingMerchantMessage).catch((e) => console.error(e))
                }

                if (envService.getBoolean("ENABLE_RPG_IMAGES") && duckHuntService.getItemsToInspect(channel)?.length > 0) {
                    const itemToInspect = duckHuntService.getItemsToInspect(channel).shift()
                    const buff = await utils.generatePicture(utils.sanitize(itemToInspect.name || 'undefined', false), 1000, 6, false)
                    if (buff) {
                        const embed = new MessageEmbed()
                            .setColor('#ffff66')
                            .setTitle(`Identified item!`)
                            .setDescription(`${itemToInspect.name} (${itemToInspect.rarity} ${itemToInspect.type})`)

                        itemToInspect.image = buff.toString('base64')
                        const buff2 = new Buffer.from(itemToInspect.image, "base64")
                        const imgOriginal = await sharp(Buffer.from(buff2, 'binary'))
                        const im = await imgOriginal.resize(160, 160, {kernel: sharp.kernel.nearest})
                        const messageAttachment = new MessageAttachment(await im.toBuffer(), "output.png")
                        embed.attachFiles([messageAttachment])
                        await channels[channel].send(embed).catch((e) => console.error(e))
                    }
                }

                if (pawn && (Date.now() - pawn.createdAt < 1000 * envService.getRpgRespawnCoolDown())) continue
                if (!swarmMode && (pawnService.lastPawnKilledAt[channel] && (Date.now() - pawnService.lastPawnKilledAt[channel] < 1000 * envService.getRpgSpawnCoolDown()))) continue
                if (swarmMode && pawn?.alive) continue

                let difficulty = swarmMode ? swarm.difficulty : null

                let spawnMessage
                if (swarmMode) {
                    spawnMessage = await duckHuntService.swarm(channel, difficulty, swarm.name || null)
                } else {
                    spawnMessage = await duckHuntService.spawn(channel, difficulty, null)
                    spawnMessage.newPawn.alive = false

                    const args = [
                        {name: "name", value: spawnMessage.newPawn.name},
                        {name: "difficulty", value: spawnMessage.newPawn.difficulty},
                        {name: "item"},
                        {name: "type"},
                        {name: "rarity"},
                    ]
                    const lootedItem = await generatorService.newWorkflow("loot", args)

                    spawnMessage.newPawn.loot = {
                        name: lootedItem.item,
                        type: lootedItem.type,
                        rarity: lootedItem.rarity
                    }

                    if (envService.getBoolean("ENABLE_RPG_IMAGES")) {
                        const buff = await utils.generatePicture(utils.sanitize(spawnMessage.newPawn.loot.name, false), 1000, 6, false)
                        if (buff) {
                            spawnMessage.newPawn.loot.image = buff.toString('base64')
                        }
                    }

                    spawnMessage.newPawn.alive = true
                }

                const m = await channels[channel].send(spawnMessage?.message).catch((e) => console.error(e))

                if (m && spawnMessage && spawnMessage.pushIntoHistory) {
                    historyService.pushIntoHistory(spawnMessage.pushIntoHistory[0], spawnMessage.pushIntoHistory[1], spawnMessage.pushIntoHistory[2], m?.id)
                }

                if (ttsEnabled && speak && spawnMessage.ttsMessage) {
                    speak(spawnMessage.ttsMessage, channel)
                }

                savingService.save(channel)
            }
            await utils.sleep(1000)
        }
    }
}

mainRpgLoop().then()

//start express server
app.listen(port, () => {
    console.log(`Listening on port ${port}`)
})


export default {}