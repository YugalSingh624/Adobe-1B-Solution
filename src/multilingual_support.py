#!/usr/bin/env python3
"""
Multilingual support module for the document processing pipeline.
Provides language detection and multilingual keyword matching without affecting English performance.
"""

import re
from typing import Dict, List, Tuple, Optional

class MultilingualSupport:
    """
    Provides multilingual support for keyword detection and content analysis.
    """
    
    def __init__(self):
        # Multilingual keyword mappings
        self.multilingual_keywords = {
            'youth_indicators': {
                'en': ['college', 'student', 'friends', 'young', 'youth', 'university'],
                'fr': ['étudiant', 'étudiants', 'université', 'amis', 'jeune', 'jeunes', 'collège', 'fac'],
                'es': ['estudiante', 'universitario', 'universidad', 'amigos', 'joven', 'jóvenes', 'colegio'],
                'de': ['student', 'studenten', 'universität', 'freunde', 'jung', 'jugend', 'uni'],
                'it': ['studente', 'università', 'universitario', 'amici', 'giovane', 'giovani'],
                'pt': ['estudante', 'universidade', 'universitário', 'amigos', 'jovem', 'jovens'],
                'nl': ['student', 'universiteit', 'vrienden', 'jong', 'jeugd'],
                'hi': ['छात्र', 'विद्यार्थी', 'कॉलेज', 'विश्वविद्यालय', 'दोस्त', 'युवा', 'जवान', 'मित्र']
            },
            'family_indicators': {
                'en': ['family', 'children', 'kids', 'child', 'parents'],
                'fr': ['famille', 'enfants', 'enfant', 'gosses', 'parents'],
                'es': ['familia', 'niños', 'niño', 'hijos', 'padres'],
                'de': ['familie', 'kinder', 'kind', 'eltern'],
                'it': ['famiglia', 'bambini', 'bambino', 'figli', 'genitori'],
                'pt': ['família', 'crianças', 'criança', 'filhos', 'pais'],
                'nl': ['familie', 'kinderen', 'kind', 'ouders'],
                'hi': ['परिवार', 'बच्चे', 'बच्चा', 'बेटा', 'बेटी', 'माता-पिता', 'माँ', 'पिता', 'संतान']
            },
            'budget_indicators': {
                'en': ['budget', 'cheap', 'affordable', 'low-cost', 'inexpensive', 'economical'],
                'fr': ['budget', 'pas cher', 'abordable', 'économique', 'bon marché'],
                'es': ['presupuesto', 'barato', 'económico', 'asequible', 'bajo costo'],
                'de': ['budget', 'günstig', 'billig', 'preiswert', 'kostengünstig'],
                'it': ['budget', 'economico', 'conveniente', 'basso costo', 'poco costoso'],
                'pt': ['orçamento', 'barato', 'econômico', 'acessível', 'baixo custo'],
                'nl': ['budget', 'goedkoop', 'betaalbaar', 'voordelig'],
                'hi': ['बजट', 'सस्ता', 'किफायती', 'कम लागत', 'आर्थिक', 'कम खर्च', 'सामर्थ्य', 'कम दाम']
            },
            'luxury_indicators': {
                'en': ['luxury', 'premium', 'high-end', 'exclusive', 'upscale', 'deluxe'],
                'fr': ['luxe', 'premium', 'haut de gamme', 'exclusif', 'chic'],
                'es': ['lujo', 'premium', 'alta gama', 'exclusivo', 'elegante'],
                'de': ['luxus', 'premium', 'hochwertig', 'exklusiv', 'edel'],
                'it': ['lusso', 'premium', 'alta gamma', 'esclusivo', 'elegante'],
                'pt': ['luxo', 'premium', 'alta qualidade', 'exclusivo', 'sofisticado'],
                'nl': ['luxe', 'premium', 'hoogwaardig', 'exclusief'],
                'hi': ['लक्जरी', 'प्रीमियम', 'उच्च गुणवत्ता', 'विशेष', 'महंगा', 'शानदार', 'एक्सक्लूसिव', 'डीलक्स']
            },
            'group_indicators': {
                'en': ['group', 'groups', 'together', 'party', 'team', 'multiple'],
                'fr': ['groupe', 'groupes', 'ensemble', 'équipe', 'plusieurs'],
                'es': ['grupo', 'grupos', 'juntos', 'equipo', 'varios'],
                'de': ['gruppe', 'gruppen', 'zusammen', 'team', 'mehrere'],
                'it': ['gruppo', 'gruppi', 'insieme', 'squadra', 'diversi'],
                'pt': ['grupo', 'grupos', 'juntos', 'equipe', 'vários'],
                'nl': ['groep', 'groepen', 'samen', 'team', 'meerdere'],
                'hi': ['समूह', 'ग्रुप', 'टीम', 'साथ', 'पार्टी', 'टोली', 'दल', 'मंडली', 'कई लोग']
            },
            'business_indicators': {
                'en': ['business', 'corporate', 'conference', 'meeting', 'professional', 'work'],
                'fr': ['affaires', 'entreprise', 'conférence', 'réunion', 'professionnel', 'travail'],
                'es': ['negocio', 'empresa', 'conferencia', 'reunión', 'profesional', 'trabajo'],
                'de': ['geschäft', 'unternehmen', 'konferenz', 'meeting', 'beruflich', 'arbeit'],
                'it': ['business', 'aziendale', 'conferenza', 'riunione', 'professionale', 'lavoro'],
                'pt': ['negócio', 'empresa', 'conferência', 'reunião', 'profissional', 'trabalho'],
                'nl': ['business', 'bedrijf', 'conferentie', 'vergadering', 'professioneel', 'werk'],
                'hi': ['व्यापार', 'व्यवसाय', 'कंपनी', 'कॉन्फ्रेंस', 'मीटिंग', 'बैठक', 'व्यावसायिक', 'काम', 'कार्य']
            },
            'adventure_indicators': {
                'en': ['adventure', 'hiking', 'outdoor', 'sports', 'active', 'explore'],
                'fr': ['aventure', 'randonnée', 'extérieur', 'sport', 'actif', 'explorer'],
                'es': ['aventura', 'senderismo', 'exterior', 'deporte', 'activo', 'explorar'],
                'de': ['abenteuer', 'wandern', 'outdoor', 'sport', 'aktiv', 'erkunden'],
                'it': ['avventura', 'escursione', 'outdoor', 'sport', 'attivo', 'esplorare'],
                'pt': ['aventura', 'caminhada', 'exterior', 'esporte', 'ativo', 'explorar'],
                'nl': ['avontuur', 'wandelen', 'buiten', 'sport', 'actief', 'verkennen'],
                'hi': ['रोमांच', 'एडवेंचर', 'ट्रेकिंग', 'बाहरी', 'खेल', 'सक्रिय', 'अन्वेषण', 'घूमना', 'पर्वतारोहण']
            }
        }
        
        # Language detection patterns
        self.language_patterns = {
            'fr': ['le', 'la', 'les', 'de', 'du', 'des', 'pour', 'un', 'une', 'voyage', 'planifier', 'avec', 'dans'],
            'es': ['el', 'la', 'los', 'las', 'de', 'del', 'para', 'un', 'una', 'viaje', 'planificar', 'con', 'en'],
            'de': ['der', 'die', 'das', 'den', 'für', 'ein', 'eine', 'reise', 'planen', 'mit', 'in', 'und'],
            'it': ['il', 'la', 'lo', 'gli', 'le', 'di', 'del', 'per', 'un', 'una', 'viaggio', 'pianificare', 'con'],
            'pt': ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'para', 'um', 'uma', 'viagem', 'planejar', 'com'],
            'nl': ['de', 'het', 'een', 'van', 'voor', 'in', 'met', 'reis', 'plannen', 'en'],
            'hi': ['के', 'की', 'का', 'में', 'से', 'को', 'है', 'हैं', 'और', 'या', 'यात्रा', 'योजना', 'साथ', 'लिए']
        }

    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        Returns 'en' for English or detected language code.
        """
        if not text:
            return 'en'
            
        text_lower = text.lower()
        
        # Count language indicators
        scores = {'en': 0}  # Default to English
        
        for lang, indicators in self.language_patterns.items():
            score = sum(1 for word in indicators if f' {word} ' in f' {text_lower} ')
            scores[lang] = score
        
        # Return language with highest score, default to English
        detected = max(scores, key=scores.get)
        return detected if scores[detected] > 0 else 'en'

    def get_multilingual_keywords(self, category: str, text: str) -> List[str]:
        """
        Get keywords for a category in the appropriate language.
        Always includes English keywords to maintain English performance.
        """
        detected_lang = self.detect_language(text)
        
        if category not in self.multilingual_keywords:
            return []
        
        # Always include English keywords for maximum compatibility
        english_keywords = self.multilingual_keywords[category]['en']
        
        # Add language-specific keywords if detected language is different
        if detected_lang != 'en' and detected_lang in self.multilingual_keywords[category]:
            native_keywords = self.multilingual_keywords[category][detected_lang]
            # Combine English and native keywords, remove duplicates
            all_keywords = list(set(english_keywords + native_keywords))
        else:
            all_keywords = english_keywords
            
        return all_keywords

    def calculate_multilingual_indicator_score(self, text: str, category: str) -> float:
        """
        Calculate indicator score using multilingual keywords.
        Maintains English performance while adding multilingual support.
        """
        keywords = self.get_multilingual_keywords(category, text)
        if not keywords:
            return 0.0
            
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Return score normalized by keyword count
        return matches / len(keywords)

    def get_enhanced_boost_factors(self, persona: str, job: str, default_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate boost factors with multilingual support.
        Enhances the existing logic without breaking English functionality.
        """
        boost_factors = default_factors.copy()
        
        # Calculate multilingual indicator scores
        indicator_scores = {}
        for category in ['youth_indicators', 'family_indicators', 'business_indicators', 
                        'adventure_indicators', 'luxury_indicators', 'budget_indicators', 'group_indicators']:
            score = self.calculate_multilingual_indicator_score(job, category)
            indicator_scores[category] = score

        # Apply dynamic boosting (same logic as before, but with multilingual detection)
        if indicator_scores['youth_indicators'] > 0.15:  # Lowered threshold for multilingual
            boost_factors.update({
                'nightlife': min(boost_factors.get('nightlife', 1.0) * 1.3, 1.5),
                'activities': min(boost_factors.get('activities', 1.0) * 1.25, 1.5),
                'dining': min(boost_factors.get('dining', 1.0) * 1.15, 1.4)
            })
            
        if indicator_scores['family_indicators'] > 0.15:
            boost_factors.update({
                'activities': min(boost_factors.get('activities', 1.0) * 1.2, 1.5),
                'culture': min(boost_factors.get('culture', 1.0) * 1.15, 1.4),
                'accommodation': min(boost_factors.get('accommodation', 1.0) * 1.2, 1.5),
                'nightlife': max(boost_factors.get('nightlife', 1.0) * 0.8, 0.7)
            })
            
        if indicator_scores['business_indicators'] > 0.15:
            boost_factors.update({
                'accommodation': min(boost_factors.get('accommodation', 1.0) * 1.3, 1.5),
                'transportation': min(boost_factors.get('transportation', 1.0) * 1.2, 1.4),
                'practical': min(boost_factors.get('practical', 1.0) * 1.25, 1.4)
            })

        if indicator_scores['luxury_indicators'] > 0.15:
            boost_factors.update({
                'accommodation': min(boost_factors.get('accommodation', 1.0) * 1.3, 1.5),
                'dining': min(boost_factors.get('dining', 1.0) * 1.3, 1.5),
                'culture': min(boost_factors.get('culture', 1.0) * 1.15, 1.4)
            })

        if indicator_scores['adventure_indicators'] > 0.15:
            boost_factors.update({
                'activities': min(boost_factors.get('activities', 1.0) * 1.4, 1.6),
                'practical': min(boost_factors.get('practical', 1.0) * 1.2, 1.4)
            })

        # Cross-factor adjustments
        if indicator_scores['budget_indicators'] > 0.1:
            boost_factors['accommodation'] = min(boost_factors.get('accommodation', 1.0) * 1.15, 1.4)

        if indicator_scores['group_indicators'] > 0.1:
            boost_factors['activities'] = min(boost_factors.get('activities', 1.0) * 1.1, 1.5)

        return boost_factors

    def calculate_multilingual_persona_boost(self, text: str, job: str) -> float:
        """
        Calculate persona-specific boost with multilingual support.
        """
        text_lower = text.lower()
        job_lower = job.lower()
        
        total_boost = 0.0
        boost_count = 0
        
        # Define scoring categories
        categories = ['budget_indicators', 'group_indicators', 'family_indicators', 
                     'luxury_indicators', 'adventure_indicators', 'business_indicators']
        
        for category in categories:
            # Check job relevance using multilingual keywords
            job_score = self.calculate_multilingual_indicator_score(job, category)
            
            if job_score > 0.1:  # Lower threshold for multilingual
                # Check content match using multilingual keywords
                content_score = self.calculate_multilingual_indicator_score(text, category)
                
                # Weight by job relevance strength
                weighted_score = content_score * min(job_score * 2, 1.0)
                total_boost += weighted_score
                boost_count += 1
        
        return (total_boost / boost_count) if boost_count > 0 else 0.0

    def get_language_info(self, text: str) -> Dict[str, any]:
        """
        Get language information for debugging and analysis.
        """
        detected_lang = self.detect_language(text)
        
        return {
            'detected_language': detected_lang,
            'is_multilingual': detected_lang != 'en',
            'confidence': 'high' if detected_lang != 'en' else 'default'
        }

# Global instance for easy access
multilingual_support = MultilingualSupport()
