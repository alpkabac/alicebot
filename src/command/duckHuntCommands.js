import {config} from "dotenv";
import Command from "./Command.js";
import duckHuntService from "../service/rpg/duckHuntService.js";
import playerService from "../service/rpg/playerService.js";
import {MessageEmbed} from "discord.js";
import pawnService from "../service/rpg/pawnService.js";

config()


const duckHuntCommands = {
    spawn: new Command(
        "Spawn",
        ["!spawn"],
        ["!spawn "],
        process.env.ALLOW_RPG_SPAWN,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            let difficulty
            let name

            const args = parsedMsg.split(';').map(s => s.trim())
            if (args.length === 2) {
                difficulty = args[0]
                name = args[1]
            } else if (args.length === 1) {
                difficulty = args[0]
            } else {
                difficulty = parsedMsg || null
            }

            return {
                message: await duckHuntService.spawn(channel, difficulty, name),
                success: true,
                deleteUserMsg: true,
                instantReply: true
            }
        },
        false
    ),
    attack: new Command(
        "Attack",
        [],
        ["!atk", "!attack", "⚔", "⚔️", ":crossed_swords:"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return await duckHuntService.attack(channel, from)
        },
        false
    ),
    loot: new Command(
        "Loot",
        ["!loot"],
        [],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return false

            const pawn = pawnService.getActivePawn(channel)
            const result = await duckHuntService.loot(channel)

            const embed = new MessageEmbed()
                .setColor('#ffff66')
                .setTitle(`Loot for ${pawn.name} (difficulty: ${pawn.difficulty.toLowerCase()}): "${result}"`)
                .setDescription(result)

            if (result) {
                return {
                    message: embed, // `[ Looted Item: ${result} ]`,
                    success: true,
                    deleteUserMsg: true,
                    instantReply: true
                }
            }
            return {error: "# Nothing to loot..."}
        },
        false
    ),
    take: new Command(
        "Grab loot",
        [],
        ["!grab", "!pick", "!take"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            const result = duckHuntService.take(channel, from, parsedMsg)
            const player = playerService.getPlayer(channel, from)

            const embed = new MessageEmbed()
                .setColor('#884422')
                .setTitle(`Player ${from} takes the item "${result.item}"`)
                .setDescription(`${from} puts the item in its backpack slot number [${player.inventory.length - 1}]`)

            if (result) {
                return {
                    message: embed, // `[ Player ${from} takes the item "${result.item}" and puts it in its backpack (slot [${player.inventory.length - 1}]) ]`,
                    success: true,
                    deleteUserMsg: true,
                    instantReply: true
                }
            } else if (result === false) {
                return {
                    error: `# Can't take item, you have something in your hands and your backpack is full. Try to \`!sell\` or \`!drop\` an item first!`,
                    instantReply: true
                }
            }
            return {
                error: "# Nothing to grab...",
                instantReply: true
            }
        },
        false
    ),
    drop: new Command(
        "Drop item",
        [],
        ["!drop"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.drop(channel, from, parsedMsg)
        },
        false
    ),
    look: new Command(
        "Look items on the floor",
        [],
        ["!look"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.look(channel)
        },
        false
    ),
    sell: new Command(
        "Sell item",
        [],
        ["!sell"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.sell(channel, from, parsedMsg)
        },
        false
    ),
    equipWeapon: new Command(
        "Equip Weapon",
        [],
        ["!equipWeapon", "!equip Weapon", "!equip weapon", "!equipW", "!equip W"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.equipWeapon(channel, from, parsedMsg.trim())
        },
        false
    ),
    equipArmor: new Command(
        "Equip armor",
        [],
        ["!equipArmor", "!equip Armor", "!equip armor", "!equipAr", "!equip Ar", "!equip ar"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.equipArmor(channel, from, parsedMsg.trim())
        },
        false
    ),
    equipAccessory: new Command(
        "Equip accessory",
        [],
        ["!equipAccessory", "!equip Accessory", "!equip accessory", "!equipAcc", "!equip Acc", "!equip acc", "!equipAc", "!equip Ac", "!equip ac"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.equipAccessory(channel, from, parsedMsg.trim())
        },
        false
    ),
    unequipWeapon: new Command(
        "Unequip Weapon",
        [],
        ["!unequipWeapon", "!unequip Weapon", "!unequipW", "!unequip W"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.unequipWeapon(channel, from)
        },
        false
    ),
    unequipArmor: new Command(
        "Unequip armor",
        [],
        ["!unequipArmor", "!unequip Armor", "!unequip armor", "!unequipAr", "!unequip Ar", "!unequip ar"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.unequipArmor(channel, from)
        },
        false
    ),
    unequipAccessory: new Command(
        "Unequip accessory",
        [],
        ["!unequipAccessory", "!unequip Accessory", "!unequipAcc", "!unequip Acc", "!unequip acc", "!unequipAc", "!unequip Ac", "!unequip ac"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.unequipAccessory(channel, from)
        },
        false
    ),
    showInventory: new Command(
        "Show Inventory",
        ["!inventory", "!showInventory"],
        [],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.showInventory(channel, from)
        },
        false
    ),
    upgradeBackpack: new Command(
        "Upgrade Backpack",
        ["!upgradeBackpack", "!upgrade"],
        [],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.upgradeBackpack(channel, from)
        },
        false
    ),
    generateSpell: new Command(
        "Generate Spell",
        [],
        ["!spellBook", "!spellbook", "!spell"],
        process.env.ALLOW_RPG_ATTACK,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId) => {
            return duckHuntService.generateSpell(channel, parsedMsg)
        },
        false
    ),
}

duckHuntCommands.all = [
    duckHuntCommands.spawn,
    duckHuntCommands.attack,
    duckHuntCommands.loot,
    duckHuntCommands.take,
    duckHuntCommands.drop,
    duckHuntCommands.look,
    duckHuntCommands.sell,
    duckHuntCommands.equipWeapon,
    duckHuntCommands.equipArmor,
    duckHuntCommands.equipAccessory,
    duckHuntCommands.unequipWeapon,
    duckHuntCommands.unequipArmor,
    duckHuntCommands.showInventory,
    duckHuntCommands.upgradeBackpack,
    duckHuntCommands.generateSpell,
]

export default duckHuntCommands