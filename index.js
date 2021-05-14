require('dotenv').config()
const axios = require('axios')
const irc = require('irc');
const options = require('./conf.json')

const ircClient = new irc.Client('chat.freenode.net', options.botName, {
    channels: [options.channel],
    username: process.env.USERNAME,
    realName: process.env.REALNAME,
    password: process.env.PASSWORD
});

const channelHistory = []
const individualHistories = {}

/**
 * Load translations, you can use the different files for different languages
 */
const translations = require('./translations/default.json')
const botMemory = require(`./translations/${options.botName}/default.json`)

/**
 * Makes the bot send a message randomly if nobody talks
 * Is limited to
 * @return {number} id of the setTimeout
 */
const minBotMessageInterval = 1000 * 60 * options.minBotMessageIntervalInMinutes
const maxBotMessageInterval = 1000 * 60 * options.maxBotMessageIntervalInMinutes
let botConsecutiveMessages = 0

function startTimer() {
    return setTimeout(() => {
        if (botConsecutiveMessages < options.botMaxConsecutiveMessages) {
            generateAndSendMessage(options.channel, channelHistory, true)
            botConsecutiveMessages++
        }
        startTimer()
    }, Math.random() * (maxBotMessageInterval - minBotMessageInterval) + minBotMessageInterval)
}

let timer = startTimer()

function restartInterval() {
    clearTimeout(timer)
    timer = startTimer()
}

/**
 * Replaces the nick of the bot by the bot name
 * Helps the bot AI to keep track of who it is
 * @param msg
 * @return {*}
 */
function replaceNickByBotName(msg) {
    return msg.replace(ircClient.nick, options.botName)
}

/**
 * Updates a given history
 */
function pushIntoHistory(history, entry) {
    history.push(entry)
    while (history.length > options.maxHistory) history.shift()
}

/**
 * Makes the bot react to messages if its name is mentioned
 */
ircClient.addListener('message', function (from, to, message) {
    // Prevents PM
    if (!to.toLowerCase().includes(options.channel.toLowerCase())) return

    const msg = replaceNickByBotName(
        (message.substr(0, 1).toUpperCase() + message.substr(1)) // Uppercase the first letter
            .trim()
    )
    console.log(from + ' => ' + to + ': ' + msg);

    pushIntoHistory(channelHistory, {from, msg})

    botConsecutiveMessages = 0    // Reset consecutive message counter

    // Detects if the bot name has been mentioned, reacts if it's the case
    if (msg.toLowerCase().includes(options.botName.toLowerCase())) {
        generateAndSendMessage(options.channel, channelHistory, true)
    }
});

/**
 * Makes the bot react when someone joins the channel
 * Also triggered when the bot joins at the start
 */
ircClient.addListener('join', function (channel, nick, message) {
    console.log(`${nick} joined the channel ${options.channel}`)
    if (nick !== ircClient.nick) {
        pushIntoHistory(channelHistory, {from: nick, msg: translations.onJoin})
        generateAndSendMessage(options.channel, channelHistory, true)
    }
});

/**
 * Makes the bot react when someone leaves the channel
 */
ircClient.addListener('part', function (channel, nick) {
    console.log(`${nick} left the channel ${options.channel}`)
    pushIntoHistory(channelHistory, {from: nick, msg: translations.onPart})
    generateAndSendMessage(options.channel, channelHistory, true)
});

/**
 * Makes the bot react when someone quits the channel
 */
ircClient.addListener('quit', function (nick) {
    console.log(`${nick} quit the channel ${options.channel}`)
    pushIntoHistory(channelHistory, {from: nick, msg: translations.onQuit})
    generateAndSendMessage(options.channel, channelHistory, true)
});

/**
 * Makes the bot react when someone is kicked from the channel
 */
ircClient.addListener('kick', function (channel, nick, by, reason) {
    console.log(`${nick} was kicked from the channel ${options.channel} by ${by}, reason: ${reason}`)
    pushIntoHistory(channelHistory, {
        from: nick,
        msg: translations.onKick.replace("${by}", by).replace("${reason}", reason)
    })
    generateAndSendMessage(options.channel, channelHistory, true)
});

/**
 * Makes the bot react when someone makes an action on the channel
 */
ircClient.addListener('action', function (from, channel, text) {
    const msg = replaceNickByBotName((text.substr(0, 1).toUpperCase() + text.substr(1)).trim())
    console.log(`${from} performed an action on ${options.channel}: ${msg}`)
    if (options.channel === ircClient.nick) {
        pushIntoHistory(individualHistories[from], {from, msg: translations.onAction.replace("${text}", msg)})
        generateAndSendMessage(from, individualHistories[from], true)
    } else {
        pushIntoHistory(channelHistory, {from, msg: translations.onAction.replace("${text}", msg)})
        generateAndSendMessage(options.channel, channelHistory, true)
    }
});

/**
 * Makes the bot react to every PM
 */
ircClient.addListener('pm', function (from, message) {
    if (!individualHistories[from]) individualHistories[from] = []  // Init individual history
    const msg = replaceNickByBotName((message.substr(0, 1).toUpperCase() + message.substr(1)).trim())
    console.log(from + ' => ' + options.botName + ': ' + msg);
    pushIntoHistory(individualHistories[from], {from, msg})
    generateAndSendMessage(from, individualHistories[from], true)
});

/**
 * Sends message using history and bot introduction as a prompt to generate the message
 * Retries until fulfillment
 * @param to the channel or nick
 * @param history of the conversation
 * @param usesIntroduction if it should use the introduction of the bot
 */
function generateAndSendMessage(to, history, usesIntroduction = false) {
    // Preparing memory by replacing placeholders
    const introduction = botMemory.introduction.map((e) => {
        return {
            from: e.from.replace("${botName}", options.botName),
            msg: e.msg
        }
    })

    // Preparing the prompt
    const prompt = (usesIntroduction ? introduction : [])   // Load introduction if needed
            .concat(
                history.slice(-options.maxHistory)                  // Concat the last X messages from history
            )
            .map((msg) => `${msg.from}: ${msg.msg}`)        // Formatting the line
            .join("\n")                                     // Concat the array into multiline string
        + "\n" + options.botName + ":"                              // Add the options.botName so the AI knows it's its turn to speak

    // Tries to generate a message until it works
    generateMessage(prompt, (message, err) => {
        message = message.trim()
        if (message && !err) {
            const messages = message.split("\n")
            for(let m of messages){
                history.push({from: options.botName, msg: m})
            }
            console.log(options.botName + ' => ' + to + ': ' + message);
            ircClient.say(to, message.trim());
            restartInterval()
        } else {
            generateAndSendMessage(to, history, usesIntroduction)
        }
    })
}

/**
 * Tries to generate a message with the AI API
 * @param prompt to feed to the AI
 * @param callback (aiMessage, err) => null either aiMessage or err if the message was null or empty after processing
 */
async function generateMessage(prompt, callback = (aiMessage, err) => null) {
    const data = {
        prompt,
        nb_answer: 1,   // Keep at 1, AI API allows to generate multiple answers per prompt but we only use the first
        raw: false,     // Keep at false
        generate_num: options.generate_num, // Number of token to generate
        temp: options.temp                  // Temperature
    }

    axios.post("http://localhost:5000/prompt", data)
        .then((result) => {
            const answer = result.data[0].startsWith(options.botName + ": ") ?  // Remove starting bot name if present
                result.data[0].slice((options.botName + ": ").length)
                : result.data[0]

            // Remove everything from the output that is not something that the bot says itself
            const parsedAnswer = answer
                .split(`${options.botName} :`)
                .join("\n")
                .split(`${options.botName}:`)
                .join("\n")
                .split(/([a-zA-Z0-9-_]+ :)/)[0]           // Remove text after first "nick: "
                .split(/([a-zA-Z0-9-_]+:)/)[0]           // Remove text after first "nick:"
                .replace(/  +/g, ' ')      // Remove double spaces
                .replace(/\n /g, '\n')      // Remove double spaces
                .split("\n")
                .slice(0, 2)
                .join("\n")
            if (parsedAnswer) {
                callback(parsedAnswer)
            } else {
                callback(null, true)
            }
        })
        .catch((err) => {
            console.log(err)
            callback(null, true)
        })
}