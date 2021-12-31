import {config} from "dotenv";

config()
import Command from './Command.js'
import MuteService from '../service/muteService.js'

const muteCommands = {
    mute: new Command(
        "Mute",
        ["!mute"],
        [],
        process.env.ALLOW_MUTE,
        (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId, client, attachmentUrl) => {
            MuteService.setChannelMuteStatus(channel, true)
            return {success: true}
        }
    ),
    unmute: new Command(
        "Unmute",
        ["!unmute"],
        [],
        process.env.ALLOW_MUTE,
        (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId, client, attachmentUrl) => {
            MuteService.setChannelMuteStatus(channel, false)
            return {success: true}
        }
    )
}

muteCommands.all = [
    muteCommands.mute,
    muteCommands.unmute
]

export default muteCommands