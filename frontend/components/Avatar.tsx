import { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { createAvatar } from '@dicebear/core';
import { avataaars } from '@dicebear/collection';

interface AvatarProps {
    emotion: string;
    isThinking: boolean;
}

export default function Avatar({ emotion, isThinking }: AvatarProps) {
    const avatarUri = useMemo(() => {
        const getOptions = () => {
            switch (emotion?.toLowerCase()) {
                case 'joy': return {
                    mouth: ['smile', 'laughing'],
                    eyes: ['happy', 'wink'],
                    eyebrows: ['default']
                };
                case 'sadness': return {
                    mouth: ['sad', 'pucker'],
                    eyes: ['squint', 'closed'],
                    eyebrows: ['sadConcerned']
                };
                case 'anger': return {
                    mouth: ['grimace', 'serious'],
                    eyes: ['angry', 'side'],
                    eyebrows: ['angry']
                };
                case 'fear': return {
                    mouth: ['concerned', 'serious'],
                    eyes: ['surprised', 'eyeRoll'],
                    eyebrows: ['raisedExcitedNatural']
                };
                case 'love': return {
                    mouth: ['smile', 'tongue'],
                    eyes: ['hearts', 'happy'],
                    eyebrows: ['defaultNatural']
                };
                default: return {
                    mouth: ['smile', 'default'],
                    eyes: ['default', 'eyeRoll'],
                    eyebrows: ['default']
                };
            }
        };

        const options = getOptions() as any;
        const avatar = createAvatar(avataaars, {
            seed: 'SereneMind',
            backgroundColor: ['b6e3f4', 'c0aede', 'd1d4f9'],
            backgroundType: ['gradientLinear'],
            mouth: options.mouth,
            eyes: options.eyes,
            eyebrows: options.eyebrows,
            clothing: ['graphicShirt', 'hoodie', 'overall'],
            top: ['shortHair', 'longHiTop', 'shaggyMullet', 'shortCurly'] as any,
            hairColor: ['2e1505', '4e1b0b', 'b58143'],
            accessories: ['round', 'prescription01'],
            clothesColor: ['262e33', '65c9ff', '5199e4'],
        });

        return avatar.toDataUri();
    }, [emotion]);

    return (
        <div className="relative w-32 h-32 flex items-center justify-center translate-z-0">
            {/* Thinking / Breath Animation */}
            <motion.div
                className="absolute inset-0 blur-3xl rounded-full bg-indigo-500/20"
                animate={{
                    scale: isThinking ? [1, 1.3, 1] : [1, 1.1, 1],
                    opacity: [0.3, 0.6, 0.3]
                }}
                transition={{ duration: 4, repeat: Infinity }}
            />

            {/* Main Avatar Display */}
            <motion.div
                className="relative w-full h-full rounded-full overflow-hidden border-2 border-white/10 shadow-2xl bg-slate-900"
                initial={false}
                animate={{
                    scale: isThinking ? 0.95 : 1,
                    rotate: isThinking ? [0, -2, 2, 0] : 0
                }}
                transition={{
                    scale: { type: "spring", stiffness: 300, damping: 20 },
                    rotate: { repeat: Infinity, duration: 2 }
                }}
            >
                <AnimatePresence mode="wait">
                    <motion.img
                        key={emotion}
                        src={avatarUri}
                        alt="SereneMind AI Avatar"
                        className="w-full h-full object-cover"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 1.1 }}
                        transition={{ duration: 0.4 }}
                    />
                </AnimatePresence>

                {/* Glassmorphic Overlay */}
                <div className="absolute inset-0 pointer-events-none bg-gradient-to-tr from-white/5 to-transparent mix-blend-overlay" />
            </motion.div>

            {/* Thinking Indicator Dots */}
            {isThinking && (
                <div className="absolute -bottom-2 flex gap-1 bg-slate-800 px-2 py-1 rounded-full border border-slate-700 shadow-lg">
                    {[0, 1, 2].map((i) => (
                        <motion.div
                            key={i}
                            className="w-1.5 h-1.5 bg-indigo-400 rounded-full"
                            animate={{ opacity: [0.3, 1, 0.3] }}
                            transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                        />
                    ))}
                </div>
            )}
        </div>
    );
}
